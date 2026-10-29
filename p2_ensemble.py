"""Part 2, task 4: soft-voting ensemble of the Part-1 models + Random Forest / AdaBoost on penultimate-layer CNN features.
Usage: python p2_ensemble.py"""
import glob, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lib import *
from lib2 import *
from nn_lib import predict

df = load_split(); y = df.label.values; sp = df.split.values
va, te, tr = sp == "val", sp == "test", sp == "train"
members = {}
for exp in ("arogya", "arogya_noskip", "arogya_nose", "transfer_resnet18", "transfer_mobilenet_v3_small", "transfer_efficientnet_b0"):
    for s in SEEDS:
        if (RESULTS / "preds" / f"{exp}_s{s}.npz").exists():
            members[f"{exp}_s{s}"] = stored_probs(exp, s)[0]
if len(members) < 2: raise SystemExit("need >=2 trained members (run train.py)")
def row(name, p):
    mv, mt = compute_metrics(y[va], p[va]), compute_metrics(y[te], p[te])
    return dict(model=name, val_pr_auc=mv["pr_auc"], val_roc_auc=mv["roc_auc"], test_pr_auc=mt["pr_auc"], test_roc_auc=mt["roc_auc"],
                test_bal_acc=mt["bal_acc"], test_recall=mt["recall"], test_specificity=mt["specificity"])
rows = [row(k, v) for k, v in members.items()]
groups = {"ensemble: all members": list(members)}
for exp in ("arogya", "transfer_resnet18"):
    ks = [k for k in members if k.startswith(exp + "_s")]
    if len(ks) > 1: groups[f"ensemble: {exp} x{len(ks)} seeds"] = ks
ens = {}
for g, ks in groups.items():
    ens[g] = np.mean([members[k] for k in ks], 0); rows.append(row(g, ens[g]))
# features from the first ArogyaNet seed (chosen by validation AUC) -> RF / AdaBoost
fi = final_info(); m, cfg, st = load_model(fi["name"], fi["best_seed"]); dfa, X, M = load_split(cfg["res"], cfg.get("prep", "none"))
E = predict(m, X, None, embed=True)
for name, clf in (("RandomForest on CNN embedding", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=GLOBAL_SEED)),
                  ("AdaBoost on CNN embedding", AdaBoostClassifier(n_estimators=200, random_state=GLOBAL_SEED))):
    clf.fit(E[tr], y[tr]); p = clf.predict_proba(E)[:, 1]; rows.append(row(name, p))
res = pd.DataFrame(rows); res.to_csv(TABLES / "p2_ensembles.csv", index=False)
# diversity: correlation of the members' error indicators on val
err = np.stack([((members[k][va] >= .5) != (y[va] == 1)).astype(float) for k in members])
corr = np.corrcoef(err)[np.triu_indices(len(members), 1)] if len(members) > 1 and err.std(1).min() > 0 else np.array([np.nan])
cm_ = float(np.nanmean(corr)) if np.isfinite(corr).any() else float('nan'); corr_txt = 'n/a (a member made no/all errors)' if np.isnan(cm_) else f'{cm_:.2f}'
single = res[res.model.isin(members)]; best_single = single.loc[single.val_pr_auc.idxmax()]
ens_row = res[res.model == "ensemble: all members"].iloc[0]
md = ["## Part 2.4 Ensembles\n", "All numbers threshold-free (PR/ROC-AUC) plus balanced accuracy / recall / specificity at 0.5; the *best member* is chosen on validation PR-AUC.\n",
      md_table(res[["model", "val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc", "test_bal_acc", "test_recall", "test_specificity"]]), "",
      f"Best single member (by val PR-AUC): `{best_single.model}`. Mean pairwise correlation of members' validation errors: {corr_txt}.\n"]
if ens_row.val_pr_auc > best_single.val_pr_auc + 1e-4 and ens_row.test_pr_auc >= best_single.test_pr_auc - 1e-4:
    md.append("**The all-members soft vote beats its best member** on both splits, as expected when members make partly different errors.\n")
else:
    md.append("**The soft vote does not beat its best member.** Averaging only helps when errors are diverse; here the members are trained on the same images with the same labels, "
              f"so their mistakes overlap (mean error correlation {corr_txt}) and the remaining errors are the hard/ambiguous or mislabeled cases that no member gets right, "
              "while averaging in weaker members (e.g. ablated variants) pulls the mean score towards theirs. The comparison is also within the seed-to-seed noise reported in Part 1.\n")
md.append("Caveat: RF/AdaBoost are fitted on *training-set* embeddings, which the CNN has partly memorised, so they inherit its over-confidence on train and are usually no better than the CNN head.\n")
write_fragment("p2_ensemble", "\n".join(md))
