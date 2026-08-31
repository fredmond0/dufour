"""Dataset over the tile cache.

Features are derived on the fly rather than precomputed: the cached PNG/JPEG
tiles are ~50 kB each, whereas an 11-channel float stack is 1.4 MB, so caching
raw tiles and paying scipy's filter cost in the DataLoader workers is a ~25x
storage win and keeps an M-series GPU fed.
"""
import json, pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

from .separate import terrain_layer
from .features import stack, CHANNELS
from .copernicus import tile as cop_tile
from .fetch import map_tile
from .tiles import tile2deg

PAD = 96  # px of real neighbour context kept around the tile while filtering


class TileDataset(Dataset):
    def __init__(self, manifest, augment=True, strip=True):
        self.items = [tuple(t) for t in json.loads(pathlib.Path(manifest).read_text())]
        self.augment = augment
        self.strip = strip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z, x, y = self.items[i]
        sub = cop_tile(z, x, y, pad_px=PAD)                  # 448x448
        rgb = map_tile(z, x, y)
        if self.strip:
            # target = relief + rock drawing only; contours, water, forest and
            # route ink are the deterministic renderer's job
            rgb, _ = terrain_layer(rgb)

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
            if np.random.rand() < 0.5:
                sub = sub[:, ::-1].copy()
                rgb = rgb[:, ::-1].copy()

        lat, _ = tile2deg(x, y, z)
        feat = stack(sub, lat, z, crop=(PAD, PAD + 256, PAD, PAD + 256))
        tgt = rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        return torch.from_numpy(feat), torch.from_numpy(tgt)


N_CH = len(CHANNELS)
