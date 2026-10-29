"""Part 2, task 2: evaluation done properly.
(a) the final model at the chosen threshold on the frozen test split (confusion matrix + 8 metrics + bootstrap CIs);
(b) the whole evaluation repeated under 5-fold stratified-GROUP CV on the train+val pool (mean ± std). In every fold the threshold is
    re-chosen by the same cost rule on an inner validation slice, so the CV estimate includes the threshold-selection step.
Usage: python p2_eval.py [--res 64 --epochs 3 for smoke tests]"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lib import *
from lib2 import *
from train import data, train_cfg
from nn_lib import predict

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--res", type=int, default=None); a = ap.parse_args()
fi = final_info(); cfg = dict(fi["cfg"]); th = json.load(open(RESULTS / "threshold.json")); thr = th["threshold"]
p, sp, y = stored_probs(fi["name"], fi["best_seed"]); te = sp == "test"
m = compute_metrics(y[te], p[te], thr)
cis = {k: bootstrap_ci(y[te], p[te], lambda yy, pp, k=k: compute_metrics(yy, pp, thr)[k]) for k in ("recall", "specificity", "precision", "f1", "roc_auc")}
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["pred NORMAL", "pred PNEUMONIA"]); ax.set_yticks([0, 1], ["true NORMAL", "true PNEUMONIA"]); ax.set_title(f"test, thr={thr:.3f}")
plt.tight_layout(); plt.savefig(FIGS / "p2_confusion.png", dpi=130); plt.close()
keys = ["acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
tab_t = pd.DataFrame([dict(metric=k, value=m[k], CI95=("[%.3f, %.3f]" % cis[k]) if k in cis else "") for k in keys])

# ---- 5-fold CV
if a.epochs: cfg["epochs"] = a.epochs
if a.res: cfg["res"] = a.res
df, X, M = data(cfg["res"], cfg["prep"]); yy = df.label.values
pool = np.where(df.split.isin(["train", "val"]))[0]; rows = []
for f in range(5):
    outer = pool[df.cv_fold.values[pool] == f]; inner = pool[df.cv_fold.values[pool] != f]
    s = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=GLOBAL_SEED + f)
    tri, vai = next(s.split(inner, yy[inner], df.group.values[inner]))
    trm = np.zeros(len(df), bool); vam = np.zeros(len(df), bool); trm[inner[tri]] = True; vam[inner[vai]] = True
    model, hist, _ = train_cfg(cfg, f, df, X, M, trm, vam, verbose=False)
    pa = predict(model, X)
    t_f, cost_v, _, _ = best_threshold(yy[vam], pa[vam])
    r = compute_metrics(yy[outer], pa[outer], t_f); r.update(fold=f, thr=t_f, cost=float(expected_cost(r["recall"], 1 - r["specificity"])), n=len(outer))
    rows.append(r); print(f"fold {f}: thr {t_f:.3f} recall {r['recall']:.3f} spec {r['specificity']:.3f} auc {r['roc_auc']:.3f}", flush=True)
cv = pd.DataFrame(rows); cv.to_csv(TABLES / "p2_cv_folds.csv", index=False)
cols = ["acc", "bal_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "cost"]
cvs = pd.DataFrame([dict(metric=c, mean=cv[c].mean(), std=cv[c].std(ddof=1)) for c in cols])
md = ["## Part 2.2 Evaluation\n", f"Final model (`{fi['name']}`, seed {fi['best_seed']}) on the frozen **test** split at the camp threshold {thr:.4f}:\n",
      "![cm](figures/p2_confusion.png)\n", md_table(tab_t), "",
      f"Counts: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}. (Test prevalence of pneumonia is {y[te].mean():.2f}, so precision/accuracy here are *not* what the camp will see; see the CEO report arithmetic.)\n",
      "### 5-fold stratified-group cross-validation on the training pool (mean ± std over folds; threshold re-chosen inside each fold; 'cost' = expected cost per patient at camp prevalence)\n",
      md_table(cvs, "{:.4f}"), "",
      "**Why accuracy alone is the wrong headline number.** At the camp's ~9:1 prevalence a model that never flags anyone already scores about 90% accuracy while missing every sick child, and accuracy weights a missed pneumonia the same as a false alarm although the former costs about 20 times more. "
      f"Accuracy also moves with the prevalence of the test set ({y[te].mean():.0%} pneumonia here, {CAMP_PREVALENCE:.0%} at the camp), so the same classifier has a different accuracy at every site; sensitivity, specificity and expected cost at the deployment prevalence are the numbers that transfer.\n"]
write_fragment("p2_eval", "\n".join(md))
