# dufour

> **Generating authentic Swiss Federal Topographic maps (swisstopo 1:25,000) from raw DEMs and OpenStreetMap vectors using a hybrid neural-deterministic pipeline.**

---

## 🏔️ Background & Motivation

Switzerland’s national topographic maps—traditionally the **Landeskarte 1:25'000 (LK25 / PK25)**—are widely regarded as the gold standard of mountain cartography. Pioneered by **General Guillaume-Henri Dufour** in the 19th century and refined by master cartographers like **Eduard Imhof**, the Swiss style is characterized by:
- **Dynamic relief shading** that rotates and blends illumination across ridges to preserve slope clarity.
- **Detailed rock drawing and hachuring** depicting cliff faces, couloirs, and scree fields.
- **Subtle glacier textures and contour shading**.
- **Meticulously balanced typography, roads, and trails** specified in exact paper millimeters.

Traditional computer-generated hillshades look sterile and flat compared to hand-drawn Swiss relief. However, training a standard generative model (e.g., standard pix2pix or Diffusion) directly on raster map scans fails because the model hallucinates nonsensical roads, garbled cabin names, and text-shaped noise.

**`dufour` solves this by decoupling the map into two distinct layers:**
1. **Neural Component (Pix2Pix GAN):** Learns the complex, artistic terrain texturing—rock hachuring, cliff shading, glacier rendering, and multi-light relief—conditioned on an 11-channel DEM feature tensor.
2. **Deterministic Component (OSM + Vector Symbology):** Fetches real OpenStreetMap geometry (roads, railways with tick marks, hiking paths with dash intervals, aerial tramways with pylons, waterways, buildings) and renders them using exact, empirically measured swisstopo ink formulations and paper-millimeter specifications.

---

## 🏗️ System Architecture

```
                                  [DATA SOURCES]
                 AWS Terrarium DEM Tile            swisstopo PK25 Map Tile
               (256x256 32-bit elevation)         (256x256 1:25,000 raster)
                          │                                  │
                          ▼                                  ▼
                [dufour/fetch.py]                  [dufour/fetch.py]
                          │                                  │
                          │                          [dufour/quality.py]
                          │                         (Filter text & low relief)
                          │                                  │
                          │                          [dufour/delabel.py]
                          │                         (Inpaint letters & spot heights)
                          │                                  │
                          ▼                                  ▼
                 [dufour/features.py]                Clean Target Raster
              (11-channel multi-azimuth stack)               │
                          │                                  │
                          └──────────────┬───────────────────┘
                                         ▼
                                [dufour/dataset.py]
                             (Rotation-safe augmentation)
                                         │
                                         ▼
                                 [dufour/model.py]
                            (Pix2Pix U-Net Generator +
                              70x70 PatchGAN Discriminator)
                                         │
                                         ▼
                              Learned Swiss Terrain Base
                                         │
                                         ▼  + [dufour/osm.py] (Overpass Vector Geometry)
                                            + [dufour/legend.py] (LK25 Symbology Specs & Inks)
                                            + [dufour/frame.py] (Pixel Grid Compositor)
                                         │
                                         ▼
                         [FINAL SWISSTOPO-STYLE MAP PRODUCT]
```

---

## 🔬 Key Engineering & Cartographic Innovations

### 1. Computer Vision "Delabeling" Pipeline (`dufour/delabel.py` & `quality.py`)
A DEM has zero information about place names (e.g., *"Matterhorn"*, *"Solvaybiwak SAC"*) or spot heights (`4478`). Any lettering left in the training target forces the neural network to emit text-shaped noise smudges.
- **Morphological Differentiation:** Unlike contour lines (which are huge, elongated components) and rock hachures (which occur in dense stroke clusters), text glyphs are **small, solid, and isolated on light backgrounds**.
- **Isolation Testing:** Calculates the local dark-pixel density within a window. Glyphs surrounded by light background are flagged as text; strokes in dense hachure fields are preserved.
- **Diffusion Inpainting:** Cleans the flagged glyphs via iterative Gaussian diffusion fill before feeding them to the training dataset.

### 2. 11-Channel Terrain Conditioning Stack (`dufour/features.py`)
Rather than passing raw elevation values or a single 315° hillshade, the DEM is pre-filtered to a 45m ground cutoff (eliminating patchwork sensor resolution artifacts) and expanded into an 11-channel physics and cartography tensor:
- **`hs315`, `hs045`, `hs135`, `hs225` (4 channels):** Multi-azimuth illumination angles allowing the network to learn Eduard Imhof's principle of rotating light along opposing mountain aspects.
- **`slope`:** Normalized slope gradient.
- **`elev_abs`:** Absolute elevation scaled to 9,000m (informs treeline, snowline, and vegetation zones).
- **`elev_loc`:** Local relative elevation percentile (2nd–98th percentile).
- **`curv`:** Surface Laplacian (ridge [+] vs. valley [-] detector).
- **`aspect_s`, `aspect_c`:** $\sin(\text{aspect})$ and $\cos(\text{aspect})$ continuous directional components (avoiding $0^\circ / 360^\circ$ wrap discontinuities).
- **`rough`:** Surface roughness (standard deviation filter).

### 3. Leak-Free Regional Holdout (`scripts/01_build_dataset.py`)
Random tile splitting causes massive validation leakage because adjacent tiles share mountain faces, causing the model to memorize specific peaks.
- **Massif Isolation:** Entire mountain massifs (**Bernina** and **Uri**) are strictly isolated into the validation set, testing whether the model truly generalizes to unseen alpine topography.

### 4. Rotation-Safe Data Augmentation (`dufour/dataset.py`)
Deriving features after spatial rotation:
- Rotating finished feature stacks corrupts aspect channels (which store bearing values).
- The pipeline rotates the **raw DEM before feature derivation**, preserving strict mathematical consistency across all 11 channels.

### 5. Empirically Recovered Inks & Paper-Millimeter Symbology (`dufour/legend.py`)
- Inks were extracted via $k$-means clustering across high-alpine and valley map tiles (`scripts/palette.py`, `scripts/palette_lines.py`):
  - **Rock Contour Brown:** `#9d8c68` (`RGB 157, 140, 104`)
  - **Glacier Contour Blue:** `#7ba6bd` (`RGB 123, 166, 189`)
  - **Watercourse Line:** `#4d7f99` (`RGB 77, 127, 153`)
  - **Meadow Buff:** `#f4f3e2` (`RGB 244, 243, 226`)
  - **Forest Green:** `#c9dcb0` (`RGB 201, 220, 175`)
- Feature widths are defined in **paper millimeters at 1:25,000 scale** (e.g. primary roads $= 0.80\text{ mm}$, hiking paths $= 0.20\text{ mm}$ dashed), converted dynamically to pixel widths at any zoom level.

---

## 📁 Repository Structure

```
dufour/
├── dufour/
│   ├── dataset.py      # PyTorch Dataset over the tile cache with on-the-fly features
│   ├── delabel.py      # Morphological text detection & diffusion inpainting
│   ├── features.py     # 11-channel DEM feature extractor (multi-azimuth hillshades, curvature)
│   ├── fetch.py        # Tile fetcher with on-disk caching (swisstopo WMTS + AWS Terrarium)
│   ├── frame.py        # Multi-tile mosaic coordinate system & Web Mercator projection
│   ├── legend.py       # Exact swisstopo LK25 symbology, palette, and millimeter rules
│   ├── model.py        # Pix2Pix Generator (U-Net) & PatchGAN Discriminator with GroupNorm
│   ├── osm.py          # OpenStreetMap vector fetching via Overpass API with local caching
│   ├── quality.py      # Tile filtering rules (rejects dense text, urban centers, low relief)
│   └── tiles.py        # Web Mercator slippy-map tile math (EPSG:3857)
├── scripts/
│   ├── 01_build_dataset.py  # Harvester & regional holdout dataset builder
│   ├── 02_train.py          # Neural training harness (Pix2Pix / cGAN)
│   ├── 03_render.py         # Full-pipeline map rendering and compositor
│   ├── palette.py           # K-means recovery of swisstopo area fill ink palette
│   └── palette_lines.py     # Local-median deviation k-means for fine line feature inks
├── data/
│   ├── tiles/               # On-disk tile cache (dem/ and map/) [gitignored]
│   ├── osm/                 # On-disk Overpass vector cache [gitignored]
│   ├── train.json           # Training tile manifest
│   └── val.json             # Validation tile manifest (held-out massifs)
└── out/                     # Diagnostic outputs, delabel comparisons, and harvest logs
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch, NumPy, SciPy, Pillow

```bash
pip install torch numpy scipy pillow
```

### 1. Build / Harvest the Dataset
Harvest aligned DEM and swisstopo PK25 pairs over Swiss alpine regions with regional holdouts:

```bash
python scripts/01_build_dataset.py --zoom 15 --workers 12
```

### 2. Inspect Delabeling & Inpainting
Run diagnostic tests to view the isolation mask and diffusion inpainting on alpine tiles:

```python
from dufour.fetch import map_tile
from dufour.delabel import clean
from PIL import Image

rgb = map_tile(15, 17079, 11724) # Matterhorn
cleaned, mask = clean(rgb)

Image.fromarray(cleaned).save("out/matterhorn_cleaned.png")
```

### 3. Verify Symbology Inks
Run empirical palette recovery scripts:

```bash
python scripts/palette.py
python scripts/palette_lines.py
```
