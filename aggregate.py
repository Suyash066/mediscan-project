"""Collects results/part1/*.json -> mean±std tables, paired ablation deltas with an honest 'inside the noise' verdict,
training curves. The ONLY script (besides the final Part-2 evaluation) that reads test-split metrics.
Usage: python aggregate.py"""
import glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import *

runs = [json.load(open(f)) for f in sorted(glob.glob(str(RESULTS / "part1" / "*.json")))]
if not runs: raise SystemExit("no runs found - run train.py first")
M = ["acc", "bal_acc", "precision", "recall", "specificity", "f1", "pr_auc", "roc_auc"]
rows = []
for r in runs:
    for split in ("val", "test"):
        rows.append(dict(exp=r["exp"], seed=r["seed"], split=split, n_params=r["n_params"], best_epoch=r["best_epoch"], **{m: r[split][m] for m in M}))
long = pd.DataFrame(rows); long.to_csv(TABLES / "part1_runs_long.csv", index=False)
fmt = lambda s: f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}"
def summary(split):
    g = long[long.split == split].groupby("exp")
    t = g[M].agg(fmt); t.insert(0, "n_seeds", g.size()); t.insert(1, "params", g.n_params.first())
    return t.reset_index()
sv, st = summary("val"), summary("test"); sv.to_csv(TABLES / "part1_val_summary.csv", index=False); st.to_csv(TABLES / "part1_test_summary.csv", index=False)

def delta(base, other, split, metric):
    a = long[(long.exp == base) & (long.split == split)].set_index("seed")[metric]
    b = long[(long.exp == other) & (long.split == split)].set_index("seed")[metric]
    d = (b - a).dropna()
    return d

def verdict(d):
    if len(d) < 2: return "n/a (needs >=2 seeds)"
    m, s = d.mean(), d.std(ddof=1)
    if abs(m) <= s: return "INSIDE THE NOISE (|mean diff| <= std of diff)"
    return "better than base" if m > 0 else "worse than base"

abl = [("arogya_noskip", "D1: remove residual skips"), ("arogya_nose", "D2: remove squeeze-excitation"),
       ("arogya_meta", "two-branch (+metadata)"), ("arogya_res96", "resolution 96 vs 224"), ("arogya_augnone", "no augmentation")]
out = []
for e, label in abl:
    if not ((long.exp == e).any() and (long.exp == "arogya").any()): continue
    for split in ("val", "test"):
        for metric in ("pr_auc", "bal_acc", "recall"):
            d = delta("arogya", e, split, metric)
            if len(d): out.append(dict(change=label, split=split, metric=metric, mean_diff=d.mean(), std_diff=d.std(ddof=1) if len(d) > 1 else np.nan, n=len(d), verdict=verdict(d) if split == "test" else ""))
abl_df = pd.DataFrame(out); abl_df.to_csv(TABLES / "part1_ablations.csv", index=False)

# curves of the base model
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for r in runs:
    if r["exp"] != "arogya": continue
    h = r["history"]; ax[0].plot(h["train_loss"], label=f"train s{r['seed']}"); ax[0].plot(h["val_loss"], "--", label=f"val s{r['seed']}"); ax[1].plot(h["val_auc"], label=f"s{r['seed']}")
ax[0].set_title("ArogyaNet loss"); ax[1].set_title("val ROC-AUC"); ax[0].legend(fontsize=6); ax[1].legend(fontsize=6); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.savefig(FIGS / "p1_arogya_curves.png", dpi=130); plt.close()

lat = {r["exp"]: r["latency_ms_cpu"] for r in runs if "latency_ms_cpu" in r}
md = ["## 6-9. Neural-network results (mean ± std over seeds)\n",
      "**Validation split** (used for all decisions):\n", md_table(sv[["exp", "n_seeds", "params", "acc", "bal_acc", "recall", "specificity", "pr_auc", "roc_auc"]]), "",
      "**Frozen test split** (read once, here):\n", md_table(st[["exp", "n_seeds", "params", "acc", "bal_acc", "recall", "specificity", "pr_auc", "roc_auc"]]), "",
      "CPU latency (ms, batch 1, 4 threads, median / p95, first seed): " + "; ".join(f"{k}: {v[0]:.1f}/{v[1]:.1f}" for k, v in lat.items()) + "\n"]
if len(abl_df):
    md += ["### Paired differences vs `arogya` (same seeds; positive = change is better). Verdict rule: inside the noise if |mean diff| <= std of the per-seed diffs.\n",
           md_table(abl_df, "{:.4f}"), "",
           "With only three seeds this rule is crude; report it as such and do not claim an effect that sits inside the spread.\n"]
tab = TABLES / "arogyanet_architecture.csv"
if tab.exists():
    md += ["### ArogyaNet architecture (parameter-bearing layers; input 1x224x224)\n", md_table(pd.read_csv(tab), "{:.0f}"), ""]
write_fragment("p1_nn", "\n".join(md))
