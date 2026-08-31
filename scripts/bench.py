#!/usr/bin/env python3
"""Throughput benchmark: data pipeline vs GPU step, and epoch-time estimate."""
import pathlib, sys, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.dataset import TileDataset, N_CH
from dufour.model import Generator, Discriminator

def main():
    dev = torch.device("mps")
    ds = TileDataset("data/train.json")
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=6,
                    persistent_workers=True, prefetch_factor=4)
    it = iter(dl); next(it)
    t = time.time(); n = 0
    for _ in range(12):
        next(it); n += 8
    dps = n / (time.time() - t)
    print(f"data loader     {dps:6.1f} img/s  (6 workers)")

    G = Generator(N_CH).to(dev); D = Discriminator(N_CH).to(dev)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999)); mse = nn.MSELoss()
    x, y, k = next(it); x, y, k = x.to(dev), y.to(dev), k.to(dev)
    for _ in range(3):
        f = G(x); (((f - y).abs() * k).mean()).backward(); G.zero_grad()
    torch.mps.synchronize(); t = time.time(); S = 8
    for _ in range(S):
        f = G(x)
        rc = y * k + f.detach() * (1 - k)
        oD.zero_grad(); dr, df = D(x, rc), D(x, f.detach())
        (0.5 * (mse(dr, torch.ones_like(dr)) + mse(df, torch.zeros_like(df)))).backward(); oD.step()
        oG.zero_grad(); df = D(x, f)
        (mse(df, torch.ones_like(df)) + 80 * ((f - y).abs() * k).mean()).backward(); oG.step()
    torch.mps.synchronize()
    gps = S * 8 / (time.time() - t)
    print(f"gpu train step  {gps:6.1f} img/s  (batch 8, MPS)")
    eff = min(dps, gps); ep = len(ds) / eff
    print(f"\nbottleneck: {'data pipeline' if dps < gps else 'GPU'}")
    print(f"effective       {eff:6.1f} img/s -> {ep/60:5.1f} min/epoch  ({len(ds)} tiles)")
    for e in (20, 40, 80):
        print(f"   {e:3d} epochs = {e*ep/3600:5.1f} h")

if __name__ == "__main__":
    main()
