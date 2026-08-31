# dufour

Swiss-style topographic maps for any mountainous area, from free global data.

Named for the Dufourkarte, the 19th-century survey whose relief drawing set the
house style that swisstopo still prints.

## The architecture, and why

A single image-to-image model trained on swisstopo tiles produces something
that *looks* Swiss and is *cartographically fictional*: hallucinated roads,
text-shaped smudges, contours that do not close. So the work is split by what
each half is actually good at.

```
                 ┌───────────────────────────────────────────┐
  Copernicus ───►│ LEARNED   relief shading + rock hachures   │──┐
  GLO-30 DEM     │ (pix2pix U-Net, 16 conditioning channels)  │  │
  Sentinel-2 ───►└───────────────────────────────────────────┘  │
                                                                ├─► sheet
  DEM        ───►┌───────────────────────────────────────────┐  │
  OpenStreetMap ►│ DETERMINISTIC  contours, landcover, roads, │──┘
                 │ trails, buildings, labels, spot heights    │
                 └───────────────────────────────────────────┘
```

The network draws only what cannot be derived from data: interpretive relief
shading and Felszeichnung (rock drawing). Every feature with a right answer —
where a trail runs, how wide a road is, what a summit's elevation is — is
rendered from vector data against the LK25 legend and is exactly correct.

## Data, all free

| Layer | Source | Licence | Coverage |
|---|---|---|---|
| Training target | swisstopo LK25 raster WMTS | swisstopo OpenData, CC-BY | Switzerland |
| Elevation | Copernicus DEM GLO-30 (AWS) | free, open | global, uniform 30 m |
| Imagery | EOX s2cloudless (Sentinel-2) | CC BY-NC-SA 4.0 | global, 10 m |
| Vectors | OpenStreetMap via Overpass | ODbL | global |

All served on the standard XYZ tile grid, so every layer is pixel-aligned by
construction — no reprojection anywhere in the pipeline.

## Decisions that turned out to matter

**Copernicus, not the AWS terrain tiles.** The terrain tiles are a patchwork —
10 m 3DEP in the USA, SRTM in Patagonia, EU-DEM in the Alps. A model trained on
Swiss inputs would meet out-of-distribution terrain everywhere else. GLO-30 is
TanDEM-X derived and uniform worldwide, and visibly sharper in high mountains.

**Not swissALTI3D.** Switzerland's 2 m LiDAR is free and tempting, but training
on 2 m and inferring on 30 m is a severe domain gap: the model would learn to
read cliff microstructure that does not exist in the input anywhere else. Train
on the DEM you will infer with.

**Ridge enhancement.** A 30 m DEM shaded naively turns the Matterhorn into a
cone — its aretes are 50–100 m wide, at the sampling limit. `swiss_relief()`
unsharp-masks the surface before lighting it, blends fine/mid/coarse
illumination, and applies local contrast, which is the analytic equivalent of
what Swiss cartographers do by hand.

**Neural text detection + masked loss, not heuristic inpainting.** Classical
morphological filters struggle on high-alpine typography because isolated rock
outcrops in snow mimic text, while words on shaded cliffs blend into rock
hachures. We use a neural text detector (CRAFT via `easyocr`) to box letter
strokes and spot heights with high precision. Instead of inpainting them (which
leaves blurry discs), these regions are masked out of the training loss
entirely (`ignore_mask`), allowing the continuous DEM to guide seamless rock
drawing underneath.

**Held-out regions, not random tiles.** Neighbouring tiles share terrain, so a
random split measures memorisation. Bernina and Uri are held out whole.

## Pipeline

```bash
python3 scripts/01_build_dataset.py     # harvest + filter Swiss alpine tiles
python3 scripts/00_prefetch.py          # warm DEM / imagery caches (repeatable)
python3 scripts/04_textmasks.py         # precompute neural text masks (parallel CRAFT OCR)
python3 scripts/05_heal_dataset.py      # LaMa AI inpainting -> clean RGB targets (data/tiles/healed/)
python3 scripts/bench.py                # throughput + epoch-time estimate
python3 scripts/02_train.py --epochs 40 # resumable; writes out/ckpt/state.pt
python3 scripts/03_render.py --lat 46.5 --lon 11.3 --km 8 --out out/dolomites.png
```

`03_render.py` falls back to analytic relief when no checkpoint exists, so the
deterministic half is usable on its own. `--analytic` forces it.

## Layout

```
dufour/
  tiles.py       XYZ tile maths
  frame.py       multi-tile mosaic frame; the shared coordinate system
  fetch.py       swisstopo + terrain tile fetching, disk-cached
  copernicus.py  GLO-30 access via /vsicurl range reads
  satellite.py   Sentinel-2 tiles -> greenness / brightness / texture
  features.py    DEM -> 16 conditioning channels
  ocr.py         CRAFT-based neural lettering detection & mask caching
  delabel.py     glyph- and word-level lettering detection
  separate.py    splits the raster into terrain vs deterministic ink
  quality.py     training-tile selection
  dataset.py     torch Dataset; loads pre-healed clean tiles directly from disk
  model.py       U-Net generator + PatchGAN discriminator
  legend.py      LK25 symbology as data, widths in paper mm at 1:25'000
  render.py      deterministic vector rendering + labels
  terrain.py     DEM mosaic, contours, analytic and Swiss relief

scripts/
  00_prefetch.py      warm DEM / satellite caches
  01_build_dataset.py harvest & filter tiles with regional holdouts
  04_textmasks.py     precompute text masks for all dataset tiles (CRAFT OCR)
  05_heal_dataset.py  batch LaMa AI inpainting -> data/tiles/healed/
  02_train.py         Pix2Pix U-Net + PatchGAN training harness
  03_render.py        composite neural terrain with OSM vector overlay
  bench.py            training throughput benchmark
  compare.py          diagnostic rendering comparison
  palette.py          k-means recovery of swisstopo ink colours
```

## Known limits

- Rock drawing at 30 m is *plausible*, not surveyed: the DEM does not resolve
  individual couloirs, so stroke placement is inferred from structure and
  Sentinel-2 texture rather than measured.
- s2cloudless is CC BY-NC-SA — non-commercial only.
- OSM alignment in the high Alps is looser than Swiss cadastral survey, so
  trails can sit a few metres off a cliff edge.
- Labels are placed by a simple greedy collision test, not a real
  label-placement solver.
