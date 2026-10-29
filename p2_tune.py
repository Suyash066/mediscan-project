"""Part 2, task 3: hyper-parameter search (random search), over/under-fitting curves, early stopping, and the FINAL model.
Search space: learning rate (log 3e-4..5e-3), batch size {16,32,64}, dropout {0,.1,.3,.5}, weight decay (log 1e-6..1e-2).
Budget: --trials random configurations x 1 seed x <=12 epochs each, scored by validation PR-AUC. Then the best config is trained with 3 seeds ("final").
Usage: python p2_tune.py --trials 12"""
import argparse
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from lib import *
from train import EXPS, data, train_cfg, run
from nn_lib import predict

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=12); ap.add_argument("--search_epochs", type=int, default=12)
ap.add_argument("--final_epochs", type=int, default=None); ap.add_argument("--res", type=int, default=None); a = ap.parse_args()
base = dict(EXPS["arogya"]);
if a.res: base["res"] = a.res
df, X, M = data(base["res"], base["prep"]); y = df.label.values
tr, va = (df.split == "train").values, (df.split == "val").values
rng = np.random.default_rng(GLOBAL_SEED)
rows = []
for t in range(a.trials):
    over = dict(lr=float(10 ** rng.uniform(np.log10(3e-4), np.log10(5e-3))), bs=int(rng.choice([16, 32, 64])),
                dropout=float(rng.choice([0, .1, .3, .5])), wd=float(10 ** rng.uniform(-6, -2)), epochs=a.search_epochs, patience=4)
    model, hist, _ = train_cfg(base, 0, df, X, M, tr, va, verbose=False, **over)
    pr = float(average_precision_score(y[va], predict(model, X[va])))
    rows.append(dict(trial=t, **over, val_pr_auc=pr, best_epoch=hist["best_epoch"], best_val_loss=hist["best_val_loss"]))
    print(rows[-1], flush=True)
tun = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False); tun.to_csv(TABLES / "tuning.csv", index=False)
best = tun.iloc[0]
cfg = dict(base, lr=float(best.lr), bs=int(best.bs), dropout=float(best.dropout), wd=float(best.wd), epochs=a.final_epochs or base["epochs"])

# ---- under/over-fitting vs regularised curves (seed 0)
N = max(25, cfg["epochs"])
_, h_un, _ = train_cfg(base, 0, df, X, M, tr, va, verbose=False, dropout=0.0, wd=0.0, aug="none", epochs=N, early_stop=False, lr=cfg["lr"], bs=cfg["bs"])
_, h_re, _ = train_cfg(base, 0, df, X, M, tr, va, verbose=False, dropout=cfg["dropout"], wd=cfg["wd"], aug="std", epochs=N, early_stop=False, lr=cfg["lr"], bs=cfg["bs"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a_, h, ttl in ((ax[0], h_un, "unregularised (no dropout/wd/aug)"), (ax[1], h_re, "regularised (tuned dropout+wd, augmentation)")):
    a_.plot(h["train_loss"], label="train"); a_.plot(h["val_loss"], label="validation"); a_.axvline(h["best_epoch"] - 1, ls=":", c="k", label=f"min val loss @ep{h['best_epoch']}")
    a_.set_title(ttl); a_.set_xlabel("epoch"); a_.legend()
ax[0].set_ylabel("BCE loss"); plt.tight_layout(); plt.savefig(FIGS / "p2_overfit_curves.png", dpi=130); plt.close()
pat = EXPS["arogya"]["patience"]
def story(h, name):
    be = h["best_epoch"]; last = len(h["val_loss"]); gap_best = h["train_loss"][be - 1] - h["val_loss"][be - 1]; gap_last = h["train_loss"][-1] - h["val_loss"][-1]
    stop = min(last, be + pat)
    return (f"{name}: validation loss is lowest at epoch {be} ({h['val_loss'][be-1]:.4f}); by epoch {last} it is {h['val_loss'][-1]:.4f} "
            f"while train loss fell from {h['train_loss'][be-1]:.4f} to {h['train_loss'][-1]:.4f} (train-val gap {gap_best:+.3f} -> {gap_last:+.3f}). "
            f"Overfitting begins after epoch {be}. Early stopping (patience {pat}) would stop at epoch {stop}, saving {last-stop} of {last} epochs and "
            f"{h['val_loss'][-1]-h['val_loss'][be-1]:.4f} of validation loss.")
txt = [story(h_un, "Unregularised"), story(h_re, "Regularised")]

# ---- final model: 3 seeds, then pick the seed with the best VALIDATION ROC-AUC (never the test split)
for s in SEEDS: run("final", cfg, s, force=True)
import json
vals = {s: json.load(open(RESULTS / "part1" / f"final_s{s}.json"))["val"]["roc_auc"] for s in SEEDS}
bs = max(vals, key=vals.get)
save_json(dict(name="final", cfg=cfg, best_seed=int(bs), val_auc_by_seed=vals), RESULTS / "final_config.json")
md = ["## Part 2.3 Hyper-parameter tuning and regularisation\n",
      f"Random search, **budget: {a.trials} trials x 1 seed x <= {a.search_epochs} epochs**, ranked by validation PR-AUC (`results/tables/tuning.csv`). Top 5:\n",
      md_table(tun.head(5), "{:.4g}"), "",
      f"Chosen: lr={cfg['lr']:.2e}, batch={cfg['bs']}, dropout={cfg['dropout']}, weight decay={cfg['wd']:.1e}.\n", "![curves](figures/p2_overfit_curves.png)\n", *[t + "\n" for t in txt]]
write_fragment("p2_tune", "\n".join(md))
