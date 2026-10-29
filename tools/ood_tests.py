"""Part 2, task 7b: feed non-X-rays to predict.py. Put a real dog photo at tools/dog.jpg (or pass --image path) - none ships with the repo
because the sandbox had no internet; a synthetic colourful 'natural' image is used as a stand-in if it is missing.
A real X-ray from the test split is the positive control."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, cv2
from PIL import Image
from lib import *
from predict import assess

ap = argparse.ArgumentParser(); ap.add_argument("--image", nargs="*", default=[]); a = ap.parse_args()
rng = np.random.default_rng(0)
out = ROOT / "results" / "ood_images"; out.mkdir(parents=True, exist_ok=True)
tests = {"blank white": np.full((512, 512, 3), 255, np.uint8), "blank black": np.zeros((512, 512, 3), np.uint8),
         "gaussian noise (grey)": np.repeat(rng.integers(0, 255, (512, 512, 1), dtype=np.uint8), 3, 2)}
yy, xx = np.mgrid[0:512, 0:512]
nat = np.stack([(xx / 2).astype(np.uint8), (yy / 2).astype(np.uint8), ((xx + yy) / 4).astype(np.uint8)], -1); cv2.circle(nat, (256, 256), 120, (30, 120, 220), -1)
tests["synthetic colourful scene"] = nat
dog = [Path(p) for p in a.image] + ([ROOT / "tools" / "dog.jpg"] if (ROOT / "tools" / "dog.jpg").exists() else [])
for p in dog: tests[f"photo: {p.name}"] = np.array(Image.open(p).convert("RGB"))
df = load_split(); te = df[df.split == "test"]
for lab, name in ((0, "positive control: test NORMAL"), (1, "positive control: test PNEUMONIA")):
    r = te[te.label == lab].iloc[0]; tests[name] = np.array(Image.open(abspath(r.path)).convert("RGB"))
rows = []
for k, arr in tests.items():
    r = assess(Image.fromarray(arr)); rows.append(dict(input=k, verdict=r["label"], prob="-" if r["prob"] is None else f"{r['prob']:.3f}", note=r["note"]))
res = pd.DataFrame(rows); res.to_csv(TABLES / "p2_ood_tests.csv", index=False)
bad = res[(res.input.str.startswith(("blank", "gaussian", "synthetic", "photo"))) & res.verdict.str.contains("PNEUMONIA")]
md = ["### Out-of-distribution check (dog photo / blank / noise)\n", md_table(res), "",
      ("**FAIL:** the tool reports pneumonia on a non-X-ray: " + ", ".join(bad.input) if len(bad) else
       "All non-X-ray inputs were rejected or not reported as pneumonia. Limits: the guard is a heuristic (colour, blankness, embedding distance); grey-scale non-chest pictures that happen to land inside the training cloud cannot be excluded.") + "\n"]
write_fragment("p2_ood", "\n".join(md))
