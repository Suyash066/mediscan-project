"""Part 1, task 3: leakage-free replacement for Rohit's split.
All 5,863 images are pooled, grouped (near-duplicate clusters via pHash + parsed patient ids joined with
union-find) and split with StratifiedGroupKFold so no group straddles two splits.
Output: splits/split.csv  (columns: path,fname,label,group,split in {train,val,test}, cv_fold in 0..4 for the train+val pool)
Usage: python split.py"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from lib import *

set_seed(GLOBAL_SEED)
df = index_primary(); info = get_info(df, "primary")
H = hashes_of(info)
parent = list(range(len(df)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
i, j, d = near_dup_pairs(H, NEAR_DUP_T)
for a, b in zip(i, j): union(a, b)
pid = [patient_group(f, l) for f, l in zip(df.fname, df.label)]
first = {}
for k, p in enumerate(pid):
    if p.startswith("F:"): continue
    if p in first: union(first[p], k)
    else: first[p] = k
df["group"] = [find(k) for k in range(len(df))]
sizes = df.groupby("group").size()
print(f"{len(df)} images -> {len(sizes)} groups; largest group {sizes.max()} images ({100*sizes.max()/len(df):.1f}% of data)")
if sizes.max() > 0.05 * len(df):
    print("WARNING: a giant group exists (hash collisions on near-uniform images?). Lower NEAR_DUP_T in lib.py.")

def sgkf(idx, n, seed):
    s = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(s.split(idx, df.label.values[idx], df.group.values[idx]))
all_idx = np.arange(len(df))
rest, test = [(a, b) for a, b in sgkf(all_idx, 7, GLOBAL_SEED)][0]          # ~14.3% test
rest_idx = all_idx[rest]
tr, va = [(a, b) for a, b in sgkf(rest_idx, 6, GLOBAL_SEED + 1)][0]        # ~14.3% of total as val
df["split"] = ""
df.loc[all_idx[test], "split"] = "test"; df.loc[rest_idx[va], "split"] = "val"; df.loc[rest_idx[tr], "split"] = "train"
df["cv_fold"] = -1
pool = np.where(df.split.isin(["train", "val"]))[0]
for f, (_, te) in enumerate(sgkf(pool, 5, GLOBAL_SEED + 2)):
    df.loc[pool[te], "cv_fold"] = f
df[["path", "fname", "orig_split", "label", "group", "split", "cv_fold"]].to_csv(SPLITS / "split.csv", index=False)

# verification: no group crosses splits; no near-duplicate pair crosses splits
assert df.groupby("group").split.nunique().max() == 1
sp = df.split.values
assert not (sp[i] != sp[j]).any(), "near-duplicate pair crosses splits"
cnt = df.pivot_table(index="split", columns="label", values="path", aggfunc="count").rename(columns={0: "NORMAL", 1: "PNEUMONIA"})
cnt["total"] = cnt.sum(axis=1); cnt["pneumonia_%"] = 100 * cnt.PNEUMONIA / cnt.total
cnt = cnt.reset_index()
md = ["## 3b. The replacement split\n", md_table(cnt, "{:.1f}"), "",
      f"Groups: {len(sizes)}; largest group {sizes.max()} images. Verified by assertion: **0** near-duplicate pairs (pHash distance <= {NEAR_DUP_T}) "
      "and 0 groups cross train/val/test. 5-fold CV folds over train+val are stored in `cv_fold`.\n"]
write_fragment("p1_split", "\n".join(md))
