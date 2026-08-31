#!/usr/bin/env python3
"""Train the relief/rock generator: DEM + Sentinel-2 -> LK25 terrain layer.

Three departures from stock pix2pix, each aimed at a specific failure mode:

  masked L1        Contours, water, forest tint and lettering all live in the
                   target but are rendered deterministically later. Supervising
                   on them would teach the model to draw a second, wrong copy.
                   We mask those pixels out of the loss rather than inpainting
                   them -- diffusion fill leaves smooth grey discs, and L1 over
                   thousands of discs teaches "rock faces contain blobs".

  composited D     If the discriminator still saw contours in the real image it
                   would learn "real == has contours" and drag the generator
                   back to drawing them. So masked regions of the real image are
                   filled with the generator's own output: identical in both,
                   hence uninformative, so D judges terrain only.

  edge loss        L1 alone is minimised by the average of all plausible stroke
                   positions, i.e. grey mush. A gradient-domain term makes
                   missing hachure strokes expensive.
"""
import argparse, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.dataset import TileDataset, N_CH
from dufour.model import Discriminator, Generator

_SOBEL = None


def sobel(t):
    global _SOBEL
    if _SOBEL is None or _SOBEL[0].device != t.device:
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=t.device).view(1, 1, 3, 3)
        _SOBEL = (kx, kx.transpose(2, 3).contiguous())
    kx, ky = _SOBEL
    g = t.mean(1, keepdim=True)
    return torch.cat([F.conv2d(g, kx, padding=1), F.conv2d(g, ky, padding=1)], 1)


MIN_SUPERVISED = 0.02   # fraction of pixels


def masked_l1(a, b, keep):
    """L1 over supervised pixels only.

    The denominator is floored: some tiles (dense pasture, where the forest
    tint is deterministic) come back over 90% masked, and one that masks out
    entirely would divide by ~1e-6 and produce an enormous gradient or a NaN.
    Flooring makes such a tile contribute proportionally little instead of
    detonating the run."""
    denom = keep.sum() * a.shape[1]
    floor = MIN_SUPERVISED * keep.numel() * a.shape[1]
    return ((a - b).abs() * keep).sum() / torch.clamp(denom, min=floor)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_grid(path, x, y, fake, keep, epoch=None):
    """Four labelled columns so the grid is readable without the source."""
    from PIL import Image, ImageDraw
    def to8(t):
        return ((t.clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    cols = ["1 INPUT  ridge hillshade", "2 MASK  white=supervised",
            "3 TARGET  swisstopo", "4 OUTPUT  generated"]
    rows = []
    for i in range(min(4, x.shape[0])):
        hs = to8(x[i, 4:5].repeat(3, 1, 1))          # ridge-enhanced hillshade
        km = to8(keep[i].repeat(3, 1, 1) * 2 - 1)
        rows.append(np.concatenate([hs, km, to8(y[i]), to8(fake[i])], 1))
    grid = np.concatenate(rows, 0)
    im = Image.new("RGB", (grid.shape[1], grid.shape[0] + 18), (255, 255, 255))
    im.paste(Image.fromarray(grid), (0, 18))
    d = ImageDraw.Draw(im)
    for c, name in enumerate(cols):
        d.text((c * 256 + 4, 4), name, fill=(170, 0, 0))
    if epoch is not None:
        d.text((grid.shape[1] - 60, 4), f"ep{epoch:03d}", fill=(0, 0, 170))
    im.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--l1", type=float, default=80.0)
    p.add_argument("--edge", type=float, default=12.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="out/ckpt")
    p.add_argument("--resume", default="")
    a = p.parse_args()

    dev = device(); print("device", dev, "| channels", N_CH, flush=True)
    tr = TileDataset("data/train.json", augment=True)
    va = TileDataset("data/val.json", augment=False)
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    dl = torch.utils.data.DataLoader(
        tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
        drop_last=True, persistent_workers=a.workers > 0, prefetch_factor=4)
    vl = torch.utils.data.DataLoader(va, batch_size=a.batch, num_workers=2)

    G = Generator(N_CH).to(dev)
    D = Discriminator(N_CH).to(dev)
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))
    mse = nn.MSELoss()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    best = 1e9; step = 0; ep0 = 0

    # Full-state checkpointing: G, D and both optimiser states. Resuming with a
    # fresh discriminator throws away the adversary the generator was tuned
    # against and the losses spike for several hundred steps.
    ck = out / "state.pt"
    if a.resume or ck.exists():
        path = pathlib.Path(a.resume) if a.resume else ck
        st = torch.load(path, map_location=dev)
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        oG.load_state_dict(st["oG"]); oD.load_state_dict(st["oD"])
        ep0 = st.get("epoch", 0) + 1; best = st.get("best", 1e9); step = st.get("step", 0)
        print(f"resumed from {path} at epoch {ep0} (best val {best:.4f})", flush=True)

    for ep in range(ep0, a.epochs):
        G.train(); t0 = time.time(); agg = np.zeros(4); nb = 0
        for x, y, keep in dl:
            x, y, keep = x.to(dev), y.to(dev), keep.to(dev)
            fake = G(x)

            # --- D on composited real: masked pixels identical in both -------
            real_c = y * keep + fake.detach() * (1 - keep)
            oD.zero_grad(set_to_none=True)
            dr, df = D(x, real_c), D(x, fake.detach())
            lD = 0.5 * (mse(dr, torch.ones_like(dr)) + mse(df, torch.zeros_like(df)))
            lD.backward(); oD.step()

            # --- G -----------------------------------------------------------
            oG.zero_grad(set_to_none=True)
            df = D(x, fake)
            l_adv = mse(df, torch.ones_like(df))
            l_l1 = masked_l1(fake, y, keep)
            l_edge = masked_l1(sobel(fake), sobel(y), keep)
            (l_adv + a.l1 * l_l1 + a.edge * l_edge).backward(); oG.step()

            agg += [lD.item(), l_adv.item(), l_l1.item(), l_edge.item()]
            nb += 1; step += 1
            if step % 100 == 0:
                d_, g_, l_, e_ = agg / nb
                print(f"  ep{ep:03d} s{step:06d} D {d_:.3f} adv {g_:.3f} "
                      f"L1 {l_:.4f} edge {e_:.4f} "
                      f"({nb*a.batch/(time.time()-t0):.1f} img/s)", flush=True)

        G.eval(); v = 0.0; n = 0
        with torch.no_grad():
            for x, y, keep in vl:
                x, y, keep = x.to(dev), y.to(dev), keep.to(dev)
                f_ = G(x); v += masked_l1(f_, y, keep).item(); n += 1
                if n == 1:
                    save_grid(out / f"samples/ep{ep:03d}.png", x, y, f_, keep, ep)
                if n >= 25:
                    break
        v /= max(n, 1)
        print(f"epoch {ep:03d}  train_L1 {agg[2]/max(nb,1):.4f}  val_L1 {v:.4f}"
              f"  {time.time()-t0:.0f}s", flush=True)
        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "oG": oG.state_dict(), "oD": oD.state_dict(),
                    "epoch": ep, "best": best, "step": step}, out / "state.pt")
        torch.save(G.state_dict(), out / "g_latest.pt")
        if v < best:
            best = v; torch.save(G.state_dict(), out / "g_best.pt")
            print(f"  * new best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
