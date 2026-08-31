"""OpenStreetMap vector data via Overpass, cached to disk.

This is the *accurate* half of the hybrid: every road, trail, building and
watercourse the final map draws comes from here as real geometry, not from a
network's imagination. The model never touches these features.
"""
import hashlib, json, pathlib, time, urllib.parse, urllib.request

CACHE = pathlib.Path("data/osm")
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Feature classes we care about, in swisstopo-legend terms.
QUERY = """
[out:json][timeout:{timeout}];
(
  way({s},{w},{n},{e})[highway];
  way({s},{w},{n},{e})[railway];
  way({s},{w},{n},{e})[aerialway];
  way({s},{w},{n},{e})[waterway];
  way({s},{w},{n},{e})[building];
  way({s},{w},{n},{e})[natural~"^(water|glacier|scree|bare_rock|wood|grassland|cliff|ridge)$"];
  way({s},{w},{n},{e})[landuse~"^(forest|meadow|farmland,?|residential|reservoir|orchard,?|vineyard)$"];
  way({s},{w},{n},{e})[power=line];
  relation({s},{w},{n},{e})[natural~"^(water|glacier)$"];
  relation({s},{w},{n},{e})[landuse=forest];
  node({s},{w},{n},{e})[natural~"^(peak|saddle)$"];
  node({s},{w},{n},{e})[place~"^(village|hamlet|town|isolated_dwelling)$"];
  node({s},{w},{n},{e})[tourism~"^(alpine_hut|wilderness_hut|viewpoint)$"];
);
out geom;
"""


def fetch(bbox, timeout=180, force=False):
    """bbox = (w, s, e, n) in degrees. Returns the raw Overpass JSON."""
    w, s, e, n = bbox
    q = QUERY.format(w=w, s=s, e=e, n=n, timeout=timeout)
    key = hashlib.sha1(q.encode()).hexdigest()[:16]
    dest = CACHE / f"{key}.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())

    last = None
    for attempt in range(4):
        for url in MIRRORS:
            try:
                data = urllib.parse.urlencode({"data": q}).encode()
                req = urllib.request.Request(
                    url, data=data,
                    headers={"User-Agent": "dufour/0.1 (cartography research)"})
                raw = urllib.request.urlopen(req, timeout=timeout + 30).read()
                obj = json.loads(raw)
                if "elements" not in obj:
                    raise ValueError("no elements")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(obj))
                return obj
            except Exception as ex:      # 429/504 are routine on public mirrors
                last = ex
        time.sleep(6 * (attempt + 1))
    raise RuntimeError(f"Overpass failed after retries: {last}")


def ways(osm):
    for el in osm.get("elements", []):
        if el.get("type") in ("way", "relation") and el.get("geometry"):
            yield el.get("tags", {}), el["geometry"]


def nodes(osm):
    for el in osm.get("elements", []):
        if el.get("type") == "node":
            yield el.get("tags", {}), el["lat"], el["lon"]


def summarize(osm):
    from collections import Counter
    c = Counter()
    for t, _ in ways(osm):
        for k in ("highway", "railway", "waterway", "building", "natural",
                  "landuse", "aerialway", "power"):
            if k in t:
                c[f"{k}={t[k]}"] += 1
    for t, _, _ in nodes(osm):
        for k in ("natural", "place", "tourism"):
            if k in t:
                c[f"node:{k}={t[k]}"] += 1
    return c
