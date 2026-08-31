"""Dataset over the tile cache.

Features are derived on the fly rather than precomputed: the cached PNG/JPEG
tiles are ~50 kB each, whereas an 11-channel float stack is 1.4 MB, so caching
raw tiles and paying scipy's filter cost in the DataLoader workers is a ~25x
storage win and keeps an M-series GPU fed.
"""
import json, pathlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .satellite import s2_padded
from .separate import deterministic_mask, ignore_mask
from .features import stack, CHANNELS
from .copernicus import tile as cop_tile
from .fetch import map_tile
from .tiles import tile2deg

PAD = 96  # px of real neighbour context kept around the tile while filtering
HEALED_DIR = pathlib.Path("data/tiles/healed")


class TileDataset(Dataset):
    def __init__(self, manifest, augment=True, strip=True, use_s2=True,
                 require_healed=True):
        self.items = [tuple(t) for t in json.loads(pathlib.Path(manifest).read_text())]
        if require_healed:
            # Train only on fully-prepared tiles. Mixing healed and un-healed
            # tiles means two different supervision regimes in one run, and the
            # un-healed path costs ~7 s/tile of live OCR, which starves the GPU.
            n0 = len(self.items)
            self.items = [t for t in self.items
                          if (HEALED_DIR / f"{t[0]}/{t[1]}/{t[2]}.png").exists()]
            print(f"[TileDataset] {len(self.items)}/{n0} tiles healed and ready")
        self.augment = augment
        self.strip = strip
        self.use_s2 = use_s2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z, x, y = self.items[i]
        sub = cop_tile(z, x, y, pad_px=PAD)                  # 448x448
        sat = s2_padded(z, x, y, pad_px=PAD) if self.use_s2 else None

        # Load pre-healed target image if available, else raw map tile
        healed_p = HEALED_DIR / f"{z}/{x}/{y}.png"
        if healed_p.exists():
            rgb = np.asarray(Image.open(healed_p).convert("RGB"))
            # LaMa heals LETTERING only. Contours, water, forest tint and route
            # ink survive healing untouched -- and the deterministic renderer
            # draws all of those itself. Supervising on them teaches the model
            # to draw a second, slightly wrong copy, so the finished sheet gets
            # doubled contours. Measured: 17.7% of a healed tile on average,
            # up to 90%. It must still be masked out of the loss.
            ign = deterministic_mask(rgb) if self.strip else np.zeros(rgb.shape[:2], bool)
        else:
            rgb = map_tile(z, x, y)
            ign = ignore_mask(rgb, z, x, y) if self.strip else np.zeros(rgb.shape[:2], bool)

        # Augment the DEM and the target *before* deriving features. Rotating a
        # finished feature stack would be wrong: hillshades rotate correctly
        # (it is equivalent to rotating the light), but aspect sin/cos store a
        # direction as a value, so they would still describe the old bearing.
        # Deriving after the rotation keeps every channel self-consistent, and
        # since the target rotates with it the model just learns a map lit from
        # whatever direction channel 0 reports.
        if self.augment:
            k = np.random.randint(4)
            if k:
                sub = np.rot90(sub, k).copy()
                rgb = np.rot90(rgb, k, (0, 1)).copy()
                ign = np.rot90(ign, k).copy()
                if sat is not None:
                    sat = np.rot90(sat, k, (0, 1)).copy()
            if np.random.rand() < 0.5:
                sub = sub[:, ::-1].copy()
                rgb = rgb[:, ::-1].copy()
                ign = ign[:, ::-1].copy()
                if sat is not None:
                    sat = sat[:, ::-1].copy()

        lat, _ = tile2deg(x, y, z)
        feat = stack(sub, lat, z, crop=(PAD, PAD + 256, PAD, PAD + 256), s2=sat)
        tgt = rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        keep = (~ign).astype(np.float32)[None]      # 1 = supervise this pixel
        return torch.from_numpy(feat), torch.from_numpy(tgt), torch.from_numpy(keep)


N_CH = len(CHANNELS)
