#!/usr/bin/env python3
"""Train the relief/rock generator: DEM feature stack -> LK25 terrain layer."""
import argparse, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dufour.dataset import TileDataset, N_CH
from dufour.model import Discriminator, Generator


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--l1", type=float, default=90.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="out/ckpt")
    p.add_argument("--resume", default="")
    a = p.parse_args()

    dev = device(); print("device", dev, flush=True)
    tr = TileDataset("data/train.json", augment=True)
    va = TileDataset("data/val.json", augment=False)
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=2)

    G = Generator(N_CH).to(dev)
    D = Discriminator(N_CH).to(dev)
    if a.resume:
        G.load_state_dict(torch.load(a.resume, map_location=dev))
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))
    mse, l1 = nn.MSELoss(), nn.L1Loss()   # LSGAN: steadier than BCE on small sets

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    step = 0
    for ep in range(a.epochs):
        G.train(); t0 = time.time(); agg = np.zeros(3); nb = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            fake = G(x)

            oD.zero_grad(set_to_none=True)
            dr = D(x, y); df = D(x, fake.detach())
            lD = 0.5 * (mse(dr, torch.ones_like(dr)) + mse(df, torch.zeros_like(df)))
            lD.backward(); oD.step()

            oG.zero_grad(set_to_none=True)
            df = D(x, fake)
            lG_adv = mse(df, torch.ones_like(df))
            lG_l1 = l1(fake, y)
            (lG_adv + a.l1 * lG_l1).backward(); oG.step()

            agg += [lD.item(), lG_adv.item(), lG_l1.item()]; nb += 1; step += 1
            if step % 50 == 0:
                d_, g_, l_ = agg / max(nb, 1)
                print(f"  ep{ep:03d} step{step:06d} D {d_:.3f} Gadv {g_:.3f} "
                      f"L1 {l_:.4f} ({nb*a.batch/(time.time()-t0):.1f} img/s)", flush=True)

        G.eval(); vl1 = 0.0; n = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vl1 += l1(G(x), y).item(); n += 1
                if n >= 25:
                    break
        print(f"epoch {ep:03d}  train_L1 {agg[2]/max(nb,1):.4f}  "
              f"val_L1 {vl1/max(n,1):.4f}  {time.time()-t0:.0f}s", flush=True)
        torch.save(G.state_dict(), out / "g_latest.pt")
        if ep % 5 == 0:
            torch.save(G.state_dict(), out / f"g_ep{ep:03d}.pt")


if __name__ == "__main__":
    main()
