"""Part 1, tasks 1-3: inventory + pixel statistics, duplicates/leakage, the 16-image validation set.
Writes figures to reports/figures, tables to results/tables and reports/fragments/p1_audit.md.
Usage: python audit.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from lib import *

set_seed(GLOBAL_SEED); log_env()
df = index_primary()
info = get_info(df, "primary")
df = pd.concat([df, info.drop(columns=["path"])], axis=1)
df["class"] = df["label"].map({0: "NORMAL", 1: "PNEUMONIA"})
md = ["## 1. Inventory and pixel statistics\n"]

# ---------------------------------------------------------------- 1. inventory
inv = df.pivot_table(index="orig_split", columns="class", values="path", aggfunc="count", fill_value=0)
inv["total"] = inv.sum(axis=1); inv["pneumonia_%"] = 100 * inv["PNEUMONIA"] / inv["total"]
inv.loc["ALL"] = [inv["NORMAL"].sum(), inv["PNEUMONIA"].sum(), inv["total"].sum(), 100 * inv["PNEUMONIA"].sum() / inv["total"].sum()]
inv = inv.reset_index().rename(columns={"orig_split": "split"}); inv.to_csv(TABLES / "inventory.csv", index=False)
md += ["Images per class per split (as shipped):\n", md_table(inv, "{:.1f}"), ""]

grey = df.groupby("class").agg(n=("path", "count"), true_gray=("true_gray", "sum"), mode_L=("mode", lambda s: (s == "L").sum()),
                               mode_RGB=("mode", lambda s: (s == "RGB").sum())).reset_index()
md += ["True greyscale (single channel, or RGB with identical channels):\n", md_table(grey), ""]

feats = ["h", "w", "aspect", "area", "mean_int", "std_int"]
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
rows = []
for a, f in zip(ax.ravel(), feats):
    for cls, c in (("NORMAL", "tab:blue"), ("PNEUMONIA", "tab:red")):
        v = df.loc[df["class"] == cls, f]
        a.hist(v, bins=40, alpha=.55, label=cls, color=c, density=True)
    a.set_title(f); a.legend()
    x, y = df.loc[df.label == 0, f], df.loc[df.label == 1, f]
    ks = stats.ks_2samp(x, y)
    pooled = np.sqrt((x.var() + y.var()) / 2)
    rows.append(dict(feature=f, normal_median=x.median(), pneumonia_median=y.median(),
                     cohens_d=(y.mean() - x.mean()) / pooled, ks_stat=ks.statistic, ks_p=ks.pvalue,
                     single_feature_AUC=max(roc_auc_score(df.label, df[f]), 1 - roc_auc_score(df.label, df[f]))))
plt.tight_layout(); plt.savefig(FIGS / "p1_pixel_stats.png", dpi=130); plt.close()
cmp_ = pd.DataFrame(rows); cmp_.to_csv(TABLES / "pixel_stats_by_class.csv", index=False)
md += ["NORMAL vs PNEUMONIA (figure `figures/p1_pixel_stats.png`; Cohen's d is PNEUMONIA minus NORMAL; "
       "`single_feature_AUC` is the ROC-AUC of that one number used as a classifier (direction chosen in its favour), 0.5 = useless):\n",
       md_table(cmp_, "{:.4g}"), ""]
strong = cmp_[(cmp_.single_feature_AUC - 0.5).abs() > 0.15]
if len(strong):
    md.append("**Automatic reading.** " + ", ".join(f"`{r.feature}` alone reaches AUC {r.single_feature_AUC:.2f}" for r in strong.itertuples()) +
              ". These numbers say nothing about lungs, so a network can score well on them without looking at pathology: "
              "they are acquisition/pre-processing differences between the two classes (a shortcut risk, tested in task 4 and task 8).\n")
else:
    md.append("**Automatic reading.** No single global statistic separates the classes strongly (|AUC-0.5|<=0.15).\n")

# ---------------------------------------------------------------- 2. duplicates and leakage
H = hashes_of(info)
exact = df.groupby("sha1").filter(lambda g: len(g) > 1)
md += ["## 2. Duplicates and leakage\n",
       f"Exact byte-identical duplicates: **{exact['sha1'].nunique()}** groups covering **{len(exact)}** files.\n"]
sens = []
for T in (0, 2, 4, 6, 8):
    i, j, d = near_dup_pairs(H, T)
    sp = df["orig_split"].values
    sens.append(dict(T=T, pairs_total=len(i), within_train=int(((sp[i] == "train") & (sp[j] == "train")).sum()),
                     within_test=int(((sp[i] == "test") & (sp[j] == "test")).sum()),
                     train_test=int((((sp[i] == "train") & (sp[j] == "test")) | ((sp[i] == "test") & (sp[j] == "train"))).sum()),
                     train_val=int((((sp[i] == "train") & (sp[j] == "val")) | ((sp[i] == "val") & (sp[j] == "train"))).sum())))
sens = pd.DataFrame(sens); sens.to_csv(TABLES / "near_dup_sensitivity.csv", index=False)
md += [f"Near-duplicate pairs by pHash Hamming-distance threshold T (T={NEAR_DUP_T} is used everywhere else):\n", md_table(sens, "{:.0f}"), ""]

i, j, d = near_dup_pairs(H, NEAR_DUP_T)
sp, lab = df["orig_split"].values, df["label"].values
cross = ((sp[i] == "train") & (sp[j] == "test")) | ((sp[i] == "test") & (sp[j] == "train"))
tr_idx = np.where(sp == "train")[0]; te_idx = np.where(sp == "test")[0]
test_with_dup = set()
for a, b in zip(i[cross], j[cross]):
    test_with_dup.add(a if sp[a] == "test" else b)
lab_conf = int((lab[i] != lab[j]).sum())
pid = df.apply(lambda r: patient_group(r.fname, r.label), axis=1)
shared_pid = set(pid[sp == "train"]) & set(pid[sp == "test"])
shared_pid = {p for p in shared_pid if not str(p).startswith("F:")}
md += [f"* Test images having at least one near-duplicate (T={NEAR_DUP_T}) in train: **{len(test_with_dup)} of {len(te_idx)} "
       f"({100*len(test_with_dup)/max(len(te_idx),1):.1f}%)**.",
       f"* Near-duplicate pairs with *different* labels (label noise): {lab_conf}.",
       f"* Patient ids (parsed from file names, heuristic) present in BOTH train and test: **{len(shared_pid)}**. "
       "The dataset ships without patient ids, so this is a lower bound: several images of one child look different to a hash but share anatomy.\n"]
# worst five cross pairs
if cross.any():
    ci, cj, cd = i[cross], j[cross], d[cross]
    import cv2
    def small(k): return cv2.resize(load_gray(df.path[k]), (64, 64), interpolation=cv2.INTER_AREA).astype(float)
    mse = np.array([np.mean((small(a) - small(b)) ** 2) for a, b in zip(ci, cj)])
    order = np.lexsort((mse, cd))[:5]
    fig, ax = plt.subplots(5, 2, figsize=(6, 14))
    for r, o in enumerate(order):
        a, b = ci[o], cj[o]
        if sp[a] != "train": a, b = b, a
        for c, k in enumerate((a, b)):
            ax[r, c].imshow(load_gray(df.path[k]), cmap="gray"); ax[r, c].axis("off")
            ax[r, c].set_title(f"{sp[k]}/{df['class'][k]}\n{df.fname[k][:28]}", fontsize=7)
        ax[r, 0].text(-0.05, 0.5, f"d={cd[o]}\nMSE={mse[o]:.1f}", transform=ax[r, 0].transAxes, ha="right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "p1_worst_pairs.png", dpi=120); plt.close()
    md.append("Worst five train/test pairs (smallest pHash distance, ties broken by pixel MSE): `figures/p1_worst_pairs.png`\n")
else:
    md.append("No train/test near-duplicate pairs were found at this threshold.\n")

# ---------------------------------------------------------------- 3. the 16-image validation set
from scipy.stats import beta
k, n = 16, 16
cp = (beta.ppf(0.025, k, n - k + 1), 1.0)
z = 1.959964; ph = k / n
wil = ((ph + z * z / (2 * n) - z * np.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)) / (1 + z * z / n), 1.0)
md += ["## 3. The validation set\n",
       f"Observed 16/16 = 100% accuracy. 95% interval for the true accuracy: Clopper-Pearson **[{cp[0]:.3f}, 1.000]**, Wilson **[{wil[0]:.3f}, 1.000]**. "
       f"The lower bound says the true accuracy could be as low as ~{100*cp[0]:.0f}%; with 8 + 8 images one error moves the estimate by 6 points, "
       "so 16 images can neither select a model nor detect overfitting.\n",
       "**Replacement (built by `split.py`)**: group-aware stratified hold-out (train/val/test, ~70/15/15; groups = near-duplicate clusters + parsed patient ids) "
       "with the test split frozen and read only by `aggregate.py`; model selection on the val split; the remaining pool (train+val) is then re-used for "
       "5-fold stratified-group CV in Part 2 (which gives mean±std). "
       "**Rejected: k-fold only** for Part 1 because (i) every experiment x 3 seeds x 5 folds multiplies compute by 5 on free Colab GPUs, and "
       "(ii) k-fold has no untouched test set: after tuning on folds, the headline number is optimistically biased. "
       "Group-awareness matters more than the choice between the two, since leakage across folds would inflate both.\n"]
save_json(dict(exact_groups=int(exact["sha1"].nunique()), test_with_dup=len(test_with_dup), n_test=int(len(te_idx)),
               label_conflicts=lab_conf, shared_patient_ids=len(shared_pid), cp_lower=float(cp[0]), wilson_lower=float(wil[0])),
          RESULTS / "audit.json")
write_fragment("p1_audit", "\n".join(md))
