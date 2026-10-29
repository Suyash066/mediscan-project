"""Part 1, tasks 4 and 5: shortcut hunt with deliberately crippled models + majority/classical baselines.
Everything is fit on split==train, hyper-parameters picked on split==val, the test split is scored once.
Usage: python baselines.py"""
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from skimage.feature import hog
from lib import *

set_seed(GLOBAL_SEED)
df, X, meta = load_split(res=224)
tr, va, te = [(df.split == s).values for s in ("train", "val", "test")]
y = df.label.values

def lung_mask(res=64):
    """Mask out the lung fields (two central ellipses covering most of the thorax); keep the frame."""
    yy, xx = np.mgrid[0:res, 0:res] / (res - 1)
    m = (((xx - 0.5) / 0.30) ** 2 + ((yy - 0.52) / 0.40) ** 2) < 1       # central thorax ellipse
    return ~m

feat = {}
feat["thumb_16x16"] = np.stack([cv2.resize(a, (16, 16), interpolation=cv2.INTER_AREA).ravel() for a in X]).astype(np.float32) / 255
mk = lung_mask(64)
feat["lungs_masked_64"] = np.stack([cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)[mk] for a in X]).astype(np.float32) / 255
feat["metadata_only"] = np.c_[meta[:, 0], meta[:, 1], np.log(meta[:, 2])]          # mean intensity, aspect, log area
feat["hog_lr"] = np.stack([hog(cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA), orientations=9,
                               pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for a in X]).astype(np.float32)

# example figure for the masked image
ex = cv2.resize(X[0], (64, 64))
vis = np.where(mk, 0, ex)
fig, a = plt.subplots(1, 3, figsize=(7, 2.5))
a[0].imshow(ex, cmap="gray"); a[0].set_title("input 64x64"); a[1].imshow(vis, cmap="gray"); a[1].set_title("lung field only (removed)")
a[2].imshow(np.where(mk, ex, 0), cmap="gray"); a[2].set_title("what the crippled model sees")
for k in a: k.axis("off")
plt.tight_layout(); plt.savefig(FIGS / "p1_shortcut_mask.png", dpi=130); plt.close()

def fit_lr(F, grid=(0.01, 0.1, 1, 10)):
    best = None
    for C in grid:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(F[tr], y[tr]); auc = roc_auc_score(y[va], m.predict_proba(F[va])[:, 1]) if len(set(y[va])) > 1 else 0
        if best is None or auc > best[0]: best = (auc, C, m)
    return best[2], best[1]

rows, preds = [], {}
for name in ("thumb_16x16", "lungs_masked_64", "metadata_only", "hog_lr"):
    if name == "metadata_only":
        m = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=GLOBAL_SEED)); m.fit(feat[name][tr], y[tr]); C = "GBM"
    else:
        m, C = fit_lr(feat[name])
    p_va, p_te = m.predict_proba(feat[name][va])[:, 1], m.predict_proba(feat[name][te])[:, 1]
    preds[name] = (p_va, p_te)
    for split, p, msk in (("val", p_va, va), ("test", p_te, te)):
        r = compute_metrics(y[msk], p); r.update(model=name, split=split, C=C)
        if split == "test":
            r["bal_acc_CI95"] = "[%.3f, %.3f]" % bootstrap_ci(y[msk], (p >= .5).astype(float), balanced_accuracy_score)
            r["roc_auc_CI95"] = "[%.3f, %.3f]" % bootstrap_ci(y[msk], p, roc_auc_score)
        rows.append(r)
# majority-class baseline
maj = int(round(y[tr].mean()))
for split, msk in (("val", va), ("test", te)):
    r = compute_metrics(y[msk], np.full(msk.sum(), float(maj))); r.update(model="majority_class", split=split, C="-"); rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(TABLES / "baselines.csv", index=False)
np.savez(RESULTS / "baseline_preds.npz", **{f"{k}_{s}": v[i] for k, v in preds.items() for i, s in enumerate(("val", "test"))})

cols = ["model", "acc", "bal_acc", "precision", "recall", "f1", "pr_auc", "roc_auc"]
t_test = res[res.split == "test"][cols + ["bal_acc_CI95", "roc_auc_CI95"]].fillna("-")
short = res[res.model.isin(["thumb_16x16", "lungs_masked_64", "metadata_only"])]
md = ["## 4. Shortcut hunt\n",
      "Crippled models (fit on train, scored on the frozen test split; chance = 0.5 balanced accuracy / ROC-AUC; majority-class accuracy = "
      f"{res[(res.model=='majority_class')&(res.split=='test')].acc.iloc[0]:.3f}):\n",
      md_table(t_test[t_test.model.isin(["thumb_16x16", "lungs_masked_64", "metadata_only"])]), "",
      "`figures/p1_shortcut_mask.png` shows the lung-masked input.\n"]
sc = res[(res.split == "test") & res.model.isin(["thumb_16x16", "lungs_masked_64", "metadata_only"])]
worst = sc.loc[sc.roc_auc.idxmax()]
md.append(f"**Automatic reading.** Strongest crippled model: `{worst.model}` with ROC-AUC {worst.roc_auc:.3f} / balanced accuracy {worst.bal_acc:.3f}. "
          + ("This is far above chance although it cannot see (or barely sees) the lung parenchyma: the label is partly readable from acquisition artefacts "
             "(global brightness/contrast, image size, borders, text markers). A CNN can use exactly the same shortcut, so high accuracy on this dataset does not prove it found pneumonia."
             if worst.roc_auc > 0.75 else
             "This is close to chance, so no strong low-level shortcut was found with these probes (it does not prove there is none).") + "\n")
md += ["## 5. Baselines (test split, threshold 0.5)\n", md_table(t_test[["model"] + cols[1:]]), ""]
write_fragment("p1_baselines", "\n".join(md))
