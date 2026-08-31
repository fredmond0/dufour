#!/usr/bin/env python3
"""Unattended chain: wait for data prep -> train -> render showcase sheets.

Runs as one background process so the whole night costs a single launch rather
than repeated polling. Every stage logs to out/overnight.log and is restartable.
"""
import json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "out/overnight.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def counts():
    out = {}
    for split in ("train", "val"):
        items = [tuple(t) for t in json.loads((ROOT / f"data/{split}.json").read_text())]
        healed = sum((ROOT / f"data/tiles/healed/{z}/{x}/{y}.png").exists()
                     for z, x, y in items)
        out[split] = (healed, len(items))
    return out


def running(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def wait_for_prep(max_hours=2.5, min_train=0.90, min_val=0.90):
    """Heal processes train.json then val.json, so val finishing means all done."""
    t0 = time.time()
    last = None
    while time.time() - t0 < max_hours * 3600:
        c = counts()
        tr, trn = c["train"]; va, van = c["val"]
        cur = (tr, va)
        if cur != last:
            log(f"prep: train {tr}/{trn} ({100*tr/trn:.1f}%)  val {va}/{van} ({100*va/van:.1f}%)")
            last = cur
        if tr >= min_train * trn and va >= min_val * van:
            log("prep complete")
            return True
        if not running("05_heal_dataset") and not running("04_textmasks"):
            log("prep jobs are no longer running")
            if tr >= 0.5 * trn and va >= 0.5 * van:
                log("enough tiles prepared; continuing anyway")
                return True
            log("too few tiles prepared; restarting healer")
            subprocess.Popen([sys.executable, "scripts/05_heal_dataset.py",
                              "--batch-size", "16"], cwd=ROOT,
                             stdout=open(ROOT / "out/heal.log", "a"),
                             stderr=subprocess.STDOUT)
            time.sleep(120)
        time.sleep(90)
    log("prep wait timed out; continuing with what exists")
    return False


SHOWCASE = [
    ("bernina_holdout", 46.3819, 9.9086, 5, "Piz Bernina - HELD OUT from training"),
    ("uri_holdout",     46.8100, 8.6100, 5, "Uri Alps - HELD OUT from training"),
    ("dolomites",       46.5300, 12.0400, 5, "Dolomites, Italy - unseen country"),
    ("sierra",          37.7340, -119.5720, 5, "Yosemite, USA - unseen continent"),
    ("chamonix",        45.9200, 6.8700, 5, "Mont Blanc, France"),
]


def main():
    log("=" * 60)
    log("overnight run starting")
    wait_for_prep()

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    log(f"launching training for {epochs} epochs")
    r = subprocess.run([sys.executable, "scripts/02_train.py",
                        "--epochs", str(epochs), "--batch", "8", "--workers", "6"],
                       cwd=ROOT, stdout=open(ROOT / "out/train.log", "a"),
                       stderr=subprocess.STDOUT)
    log(f"training exited with code {r.returncode}")

    ck = ROOT / "out/ckpt/g_best.pt"
    if not ck.exists():
        ck = ROOT / "out/ckpt/g_latest.pt"
    if not ck.exists():
        log("no checkpoint produced; stopping")
        return

    log(f"rendering showcase sheets from {ck.name}")
    for name, lat, lon, km, desc in SHOWCASE:
        dest = ROOT / f"out/showcase/{name}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"  {name}: {desc}")
        try:
            subprocess.run([sys.executable, "scripts/03_render.py",
                            "--lat", str(lat), "--lon", str(lon), "--km", str(km),
                            "--out", str(dest), "--ckpt", str(ck)],
                           cwd=ROOT, timeout=1800,
                           stdout=open(ROOT / "out/render.log", "a"),
                           stderr=subprocess.STDOUT)
        except Exception as e:
            log(f"  {name} FAILED: {e}")
    log("overnight run complete")


if __name__ == "__main__":
    main()
