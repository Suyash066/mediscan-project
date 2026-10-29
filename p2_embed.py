"""Part 2, task 5: feature-space analysis of the final model's penultimate embeddings (test split).
PCA + t-SNE coloured by class and by correct/incorrect (at the camp threshold), k-means k=2 (purity, ARI), what errors share.
Usage: python p2_embed.py"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from lib import *
from lib2 import *
from nn_lib import predict

fi = final_info(); thr = json.load(open(RESULTS / "threshold.json"))["threshold"]
m, cfg, st = load_model(fi["name"], fi["best_seed"]); df, X, M = load_split(cfg["res"], cfg.get("prep", "none"))
te = (df.split == "test").values; y = df.label.values[te]
E = predict(m, X[te], None, embed=True); p = predict(m, X[te]); wrong = (p >= thr).astype(int) != y
Z = StandardScaler().fit_transform(E)
P = PCA(2, random_state=GLOBAL_SEED).fit(Z); Zp = P.transform(Z)
T = TSNE(2, perplexity=min(30, max(5, len(Z) // 4)), random_state=GLOBAL_SEED, init="pca").fit_transform(Z)
fig, ax = plt.subplots(2, 2, figsize=(10, 9))
for row, (nm, Q) in enumerate((("PCA", Zp), ("t-SNE", T))):
    ax[row, 0].scatter(Q[y == 0, 0], Q[y == 0, 1], s=8, c="tab:blue", label="NORMAL"); ax[row, 0].scatter(Q[y == 1, 0], Q[y == 1, 1], s=8, c="tab:red", label="PNEUMONIA")
    ax[row, 0].set_title(f"{nm}: true class"); ax[row, 0].legend()
    ax[row, 1].scatter(Q[~wrong, 0], Q[~wrong, 1], s=8, c="lightgrey", label="correct"); ax[row, 1].scatter(Q[wrong, 0], Q[wrong, 1], s=22, c="black", marker="x", label="misclassified")
    ax[row, 1].set_title(f"{nm}: correct vs misclassified @thr={thr:.3f}"); ax[row, 1].legend()
plt.tight_layout(); plt.savefig(FIGS / "p2_embeddings.png", dpi=130); plt.close()
km = KMeans(2, n_init=10, random_state=GLOBAL_SEED).fit(Z); lab = km.labels_
purity = sum(max(((lab == c) & (y == k)).sum() for k in (0, 1)) for c in (0, 1)) / len(y)
ari, nmi = adjusted_rand_score(y, lab), normalized_mutual_info_score(y, lab)
# what do misclassified points have in common?
meta_te = M[te]; feats = pd.DataFrame(dict(mean_intensity=meta_te[:, 0], aspect=meta_te[:, 1], log_area=np.log(meta_te[:, 2]), score=p, true_label=y))
rows = []
for c in ("mean_intensity", "aspect", "log_area", "score"):
    a, b = feats.loc[wrong, c], feats.loc[~wrong, c]
    rows.append(dict(feature=c, misclassified_mean=a.mean() if len(a) else np.nan, correct_mean=b.mean(), std_diff=(a.mean() - b.mean()) / (feats[c].std() + 1e-9) if len(a) else np.nan))
cmp_ = pd.DataFrame(rows); cmp_.to_csv(TABLES / "p2_misclassified_vs_correct.csv", index=False)
fn, fp = int((wrong & (y == 1)).sum()), int((wrong & (y == 0)).sum())
dist = np.linalg.norm(Z - Z[y == 0].mean(0), axis=1) - np.linalg.norm(Z - Z[y == 1].mean(0), axis=1)
top = cmp_.reindex(cmp_.std_diff.abs().sort_values(ascending=False).index).head(2)
md = ["## Part 2.5 Feature-space analysis (test split, penultimate-layer embeddings)\n", "![emb](figures/p2_embeddings.png)\n",
      f"k-means with k=2 on the standardised embeddings: **purity {purity:.3f}, adjusted Rand index {ari:.3f}**, NMI {nmi:.3f} (ARI 0 = random, 1 = perfect). "
      f"First two PCs explain {100*P.explained_variance_ratio_.sum():.1f}% of the variance.\n",
      f"Misclassified at the camp threshold: **{int(wrong.sum())}** images ({fn} missed pneumonia, {fp} false alarms) out of {len(y)}. Misclassified vs correct:\n", md_table(cmp_, "{:.3f}"), "",
      f"**Automatic reading.** The largest standardised differences are in " + " and ".join(f"`{r.feature}` ({r.std_diff:+.2f} SD)" for r in top.itertuples()) +
      ". Points with a score close to the threshold sit between the two clusters in the t-SNE plot; check the figure to see whether the errors form their own small cluster (systematic, e.g. one acquisition style) or are scattered along the class boundary (genuinely ambiguous cases). Write your conclusion from the figure here.\n"]
write_fragment("p2_embed", "\n".join(md))
