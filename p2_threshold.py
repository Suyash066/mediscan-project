"""Part 2, task 1: training-side imbalance treatment vs threshold moving, and the cost-optimal threshold at camp prevalence.
Needs: train.py has produced `arogya` and `arogya_cw` (class-weighted), p2_tune.py has produced `final`.
Usage: python p2_threshold.py"""
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from lib import *
from lib2 import *

md = ["## Part 2.1 Imbalance treatment vs threshold moving, and the camp operating point\n"]
# ---------- (A) class weights change the MODEL; moving the threshold changes only the DECISION
rows = []
for s in SEEDS:
    try:
        pa, sp, y = stored_probs("arogya", s); pc, _, _ = stored_probs("arogya_cw", s)
    except FileNotFoundError:
        continue
    va = sp == "val"
    # threshold moving on the plain model that reaches the same recall as the weighted model at 0.5
    rc = compute_metrics(y[va], pc[va])["recall"]
    thr = np.sort(pa[va][y[va] == 1])[int(np.floor((1 - rc) * (y[va] == 1).sum()))] if (y[va] == 1).sum() else 0.5
    ma, mc, mm = compute_metrics(y[va], pa[va]), compute_metrics(y[va], pc[va]), compute_metrics(y[va], pa[va], thr)
    assert abs(ma["roc_auc"] - mm["roc_auc"]) < 1e-12 and abs(ma["pr_auc"] - mm["pr_auc"]) < 1e-12     # moving a threshold cannot change AUC
    sa = torch.load(MODELS / f"arogya_s{s}.pt", weights_only=False)["state"]; sc = torch.load(MODELS / f"arogya_cw_s{s}.pt", weights_only=False)["state"]
    d = float(np.sqrt(sum(((sa[k] - sc[k]) ** 2).sum().item() for k in sa if sa[k].dtype.is_floating_point)))
    rows.append(dict(seed=s, **{f"plain_{k}": ma[k] for k in ("roc_auc", "pr_auc", "recall", "specificity")},
                     **{f"cw_{k}": mc[k] for k in ("roc_auc", "pr_auc", "recall", "specificity")},
                     moved_thr=float(thr), **{f"moved_{k}": mm[k] for k in ("roc_auc", "pr_auc", "recall", "specificity")},
                     weight_L2_distance=d, score_spearman=float(spearmanr(pa[va], pc[va])[0])))
if rows:
    A = pd.DataFrame(rows); A.to_csv(TABLES / "p2_cw_vs_threshold.csv", index=False)
    mean = A.drop(columns="seed").mean()
    md += ["Validation split, mean over seeds. *plain* = unweighted loss at 0.5; *cw* = class-weighted loss at 0.5; *moved* = plain model with the threshold lowered until recall matches *cw*.\n",
           md_table(pd.DataFrame({"metric": ["roc_auc", "pr_auc", "recall", "specificity"],
                                  "plain@0.5": [mean[f"plain_{k}"] for k in ("roc_auc", "pr_auc", "recall", "specificity")],
                                  "class-weighted@0.5": [mean[f"cw_{k}"] for k in ("roc_auc", "pr_auc", "recall", "specificity")],
                                  "plain, threshold moved": [mean[f"moved_{k}"] for k in ("roc_auc", "pr_auc", "recall", "specificity")]})), "",
           f"Threshold moving leaves ROC-AUC and PR-AUC *exactly* unchanged (asserted in code; it only slides along the same curve), whereas class weights produce a different network: "
           f"mean weight distance between the plain and weighted models of the same seed = {mean['weight_L2_distance']:.2f}, rank correlation of their scores = {mean['score_spearman']:.3f}, AUCs differ ({mean['plain_roc_auc']:.4f} vs {mean['cw_roc_auc']:.4f}).\n"]
else:
    md.append("(`arogya_cw` not trained yet: run `python train.py --exps arogya_cw`)\n")

# ---------- (B) cost matrix at camp prevalence
fi = final_info(); name, seed = fi["name"], fi["best_seed"]
p, sp, y = stored_probs(name, seed)
va, te = sp == "val", sp == "test"
ds_prev = y[va].mean()
thr, cmin, tpr_v, fpr_v = best_threshold(y[va], p[va], CAMP_PREVALENCE)
# cross-check: Bayes rule under prior shift (valid only if probabilities are calibrated)
odds_shift = (CAMP_PREVALENCE / (1 - CAMP_PREVALENCE)) / (ds_prev / (1 - ds_prev))
bayes_p = COST_FP / (COST_FP + COST_FN)                      # 1/21: flag when the camp-prevalence posterior is >= 1/21
bayes_thr_ds = (bayes_p / (1 - bayes_p)) / odds_shift; bayes_thr_ds = bayes_thr_ds / (1 + bayes_thr_ds)
mt = compute_metrics(y[te], p[te], thr)
tpr_t, fpr_t = mt["recall"], 1 - mt["specificity"]
def cost_of(thr_):
    m = compute_metrics(y[te], p[te], thr_); return float(expected_cost(m["recall"], 1 - m["specificity"]))
tab = pd.DataFrame([dict(rule="flag nobody (all NORMAL)", threshold="-", recall=0.0, specificity=1.0, cost=float(expected_cost(0, 0))),
                    dict(rule="flag everybody", threshold="-", recall=1.0, specificity=0.0, cost=float(expected_cost(1, 1))),
                    dict(rule="threshold 0.5", threshold=0.5, recall=compute_metrics(y[te], p[te])["recall"], specificity=compute_metrics(y[te], p[te])["specificity"], cost=cost_of(0.5)),
                    dict(rule="cost-optimal (chosen on val)", threshold=thr, recall=mt["recall"], specificity=mt["specificity"], cost=cost_of(thr)),
                    dict(rule="Bayes rule (prior-shifted 1/21)", threshold=bayes_thr_ds, recall=compute_metrics(y[te], p[te], bayes_thr_ds)["recall"], specificity=compute_metrics(y[te], p[te], bayes_thr_ds)["specificity"], cost=cost_of(bayes_thr_ds))])
tab.to_csv(TABLES / "p2_cost_table.csv", index=False)
save_json(dict(threshold=thr, camp_prevalence=CAMP_PREVALENCE, cost_fn=COST_FN, cost_fp=COST_FP, val_cost=cmin, val_tpr=tpr_v, val_fpr=fpr_v,
               model=name, seed=int(seed), bayes_threshold=float(bayes_thr_ds)), RESULTS / "threshold.json")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for split, msk, c in (("val", va, "tab:blue"), ("test", te, "tab:orange")):
    f, t, _ = roc_curve(y[msk], p[msk]); ax[0].plot(f, t, c=c, label=f"{split} (AUC {roc_auc_score(y[msk], p[msk]):.3f})")
m = compute_metrics(y[va], p[va], thr); ax[0].scatter([1 - m["specificity"]], [m["recall"]], c="red", s=70, zorder=5, label=f"chosen thr={thr:.3f}")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC"); ax[0].legend()
for split, msk, c in (("val", va, "tab:blue"), ("test", te, "tab:orange")):
    pr, rc, _ = precision_recall_curve(y[msk], p[msk]); ax[1].plot(rc, pr, c=c, label=f"{split} @dataset prevalence {y[msk].mean():.2f}")
mm = compute_metrics(y[va], p[va], thr); ax[1].scatter([mm["recall"]], [mm["precision"]], c="red", s=70, zorder=5)
f, t, _ = roc_curve(y[va], p[va]); ppv = ppv_at_prevalence(t, f); ax[1].plot(t, ppv, "--", c="tab:green", label=f"val re-weighted to camp prevalence {CAMP_PREVALENCE:.2f}")
ax[1].scatter([mm["recall"]], [float(ppv_at_prevalence(mm["recall"], 1 - mm["specificity"]))], c="red", marker="x", s=80, zorder=5)
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-Recall (x = chosen point at camp prevalence)"); ax[1].legend(fontsize=7)
thr_c, tpr_c, fpr_c, cost_c = cost_curve(y[va], p[va]); ax[2].plot(np.clip(thr_c, 0, 1), cost_c); ax[2].axvline(thr, c="red", ls=":")
ax[2].set_xlabel("threshold"); ax[2].set_ylabel("expected cost / patient (camp, FN=20, FP=1)"); ax[2].set_title("Cost vs threshold"); ax[2].set_xscale("log")
plt.tight_layout(); plt.savefig(FIGS / "p2_roc_pr_cost.png", dpi=130); plt.close()
ppv_c = float(ppv_at_prevalence(tpr_t, fpr_t))
md += ["### Cost matrix and operating point\n",
       f"Cost: missed pneumonia = {COST_FN:.0f}, false alarm = {COST_FP:.0f}; camp prevalence {CAMP_PREVALENCE:.0%} pneumonia (9:1 NORMAL). Expected cost per patient = "
       f"{CAMP_PREVALENCE}·{COST_FN:.0f}·(1-TPR) + {1-CAMP_PREVALENCE}·{COST_FP:.0f}·FPR, with TPR/FPR measured on the validation split (they do not depend on prevalence; that is why re-weighting is legitimate). "
       f"Dataset prevalence in val is {ds_prev:.2f}, **not** the camp's.\n",
       f"**Chosen threshold = {thr:.4f}** (validation expected cost {cmin:.4f}). Prior-shift Bayes rule cross-check (needs calibrated probabilities): {bayes_thr_ds:.4f}.\n",
       "Frozen test split at the camp prevalence:\n", md_table(tab), "",
       f"At the chosen point the test recall is {mt['recall']:.3f} and specificity {mt['specificity']:.3f}; at camp prevalence the **positive predictive value is {ppv_c:.2%}**. "
       + ("Most alarms are therefore false, by arithmetic, because healthy children are 9x more common; this is acceptable only because an alarm triggers a clinician check, not a diagnosis.\n" if ppv_c < 0.5 else "\n"),
       "![roc-pr-cost](figures/p2_roc_pr_cost.png)\n"]
write_fragment("p2_threshold", "\n".join(md))
