"""Creates a TINY FAKE dataset with the same folder layout as the real ones, so the whole
pipeline can be smoke-tested in minutes:  MEDISCAN_DATA=data_synth python tools/make_synthetic_data.py
The images are noise + ellipses; no number produced from them means anything medically."""
import os
from pathlib import Path
import numpy as np, cv2
root = Path(os.environ.get("MEDISCAN_DATA", "data_synth")); rng = np.random.default_rng(0)

def fake(label, size, bright):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = rng.normal(0.25 + bright, 0.05, (h, w))
    for cx in (0.33, 0.67):
        e = ((xx - cx * w) / (0.17 * w)) ** 2 + ((yy - 0.5 * h) / (0.28 * h)) ** 2 < 1
        img[e] -= 0.12
    if label:
        for _ in range(rng.integers(2, 5)):
            cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.3, 0.7) * h
            blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.07 * min(h, w)) ** 2
            img[blob] += 0.25
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def save(path, g, rgb=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.stack([g] * 3, -1) if rgb else g)

counts = {"train": (60, 150), "val": (2, 2), "test": (30, 50)}
saved = {}
for split, (nn, npn) in counts.items():
    for lab, n, cls in ((0, nn, "NORMAL"), (1, npn, "PNEUMONIA"), ):
        for k in range(n):
            # PNEUMONIA images are smaller/brighter on average -> a planted shortcut
            size = (int(rng.integers(500, 1000)), int(rng.integers(700, 1200))) if lab else (int(rng.integers(900, 1500)), int(rng.integers(1100, 1700)))
            g = fake(lab, size, 0.08 if lab else 0.0)
            name = f"person{rng.integers(1, 400)}_bacteria_{k}.jpeg" if lab else f"IM-{rng.integers(1, 400):04d}-{k:04d}.jpeg"
            p = root / "chest_xray" / split / cls / name
            save(p, g, rgb=bool(rng.random() < 0.1)); saved[(split, cls, k)] = (p, g)
for k in range(8):   # plant near-duplicates: re-encoded copies of train images placed in test
    p, g = saved[("train", "PNEUMONIA", k)]
    save(root / "chest_xray" / "test" / "PNEUMONIA" / f"person9{k}_bacteria_dup{k}.jpeg", cv2.GaussianBlur(g, (3, 3), 0))
for lab, n, folder in ((0, 40, "Normal"), (1, 20, "Viral Pneumonia")):   # external hospital: other brightness/contrast
    for k in range(n):
        g = fake(lab, (299, 299), 0.25)
        g = np.clip(g.astype(float) * 0.7 + 60, 0, 255).astype(np.uint8)
        save(root / "covid19_radiography" / folder / "images" / f"{folder.split()[0]}-{k}.png", g)
print("synthetic data written to", root)
