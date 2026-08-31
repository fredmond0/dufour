"""Run the trained relief generator over an arbitrary frame.

Two things make this more than a forward pass.

Identical geometry.  Features are built exactly as in training -- a 256 px core
with 96 px of real neighbouring terrain around it -- so the network sees inputs
drawn from the distribution it was fitted on. Feeding it a whole 6000 px mosaic
in one go would be a different problem from the one it learned.

Overlapping windows.  A GAN's output is not perfectly consistent across a tile
boundary, so butt-joined tiles leave visible seams on smooth snowfields. We run
overlapping windows and blend them under a raised-cosine weight, which costs
about 4x the compute at 50% overlap and removes the seam entirely.
"""
import numpy as np
import torch

from .features import stack
from .model import Generator
from .satellite import s2_mosaic
from .tiles import tile2deg

WIN = 256
PAD = 96          # must match dufour.dataset.PAD


def _device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _feather(n=WIN, edge=0.5):
    """Raised-cosine weight, flat in the middle, tapering at the edges."""
    r = np.arange(n, dtype=np.float32)
    w = np.ones(n, np.float32)
    t = max(int(n * edge / 2), 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(t) + 0.5) / t)
    w[:t] = ramp
    w[-t:] = ramp[::-1]
    return np.outer(w, w)


def learned_relief(frame, dem, off, ckpt, stride=WIN // 2, batch=6,
                   device=None, use_s2=True, progress=True):
    """frame: Frame. dem/off: from dufour.terrain.dem_mosaic or copernicus.sample.

    `off` is the padding already present around the frame in `dem`; it must be
    at least PAD or edge windows lack context.
    """
    dev = _device(device)
    G = Generator(len(__import__("dufour.features", fromlist=["CHANNELS"]).CHANNELS))
    state = torch.load(ckpt, map_location="cpu")
    G.load_state_dict(state["G"] if isinstance(state, dict) and "G" in state else state)
    G.to(dev).eval()

    if off < PAD:
        raise ValueError(f"dem padding {off} px < required {PAD} px of context")
    sat = s2_mosaic(frame, pad_px=PAD) if use_s2 else None
    lat, _ = tile2deg(frame.x0, frame.y0, frame.z)

    H, W = frame.height, frame.width
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    fw = _feather()[..., None]

    ys = list(range(0, max(H - WIN, 0) + 1, stride))
    xs = list(range(0, max(W - WIN, 0) + 1, stride))
    if ys[-1] != H - WIN:
        ys.append(H - WIN)
    if xs[-1] != W - WIN:
        xs.append(W - WIN)
    coords = [(y, x) for y in ys for x in xs]

    buf, pos = [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(dev)
        with torch.no_grad():
            out = G(t).clamp(-1, 1).cpu().numpy()
        for (yy, xx), o in zip(pos, out):
            rgb = (o.transpose(1, 2, 0) + 1) * 127.5
            acc[yy:yy + WIN, xx:xx + WIN] += rgb * fw
            wsum[yy:yy + WIN, xx:xx + WIN] += fw
        buf.clear(); pos.clear()

    for n, (y, x) in enumerate(coords, 1):
        dy, dx = y + off, x + off
        sub = dem[dy - PAD:dy + WIN + PAD, dx - PAD:dx + WIN + PAD]
        s2 = None
        if sat is not None:
            sy, sx = y + PAD, x + PAD          # sat is padded by PAD, not off
            s2 = sat[sy - PAD:sy + WIN + PAD, sx - PAD:sx + WIN + PAD]
        f = stack(sub, lat, frame.z, crop=(PAD, PAD + WIN, PAD, PAD + WIN), s2=s2)
        buf.append(f); pos.append((y, x))
        if len(buf) >= batch:
            flush()
            if progress and n % (batch * 8) == 0:
                print(f"    relief {n}/{len(coords)} windows", flush=True)
    flush()

    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
