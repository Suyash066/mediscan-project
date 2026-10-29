"""Part 2, task 6: the ONE-TIME external evaluation (COVID-19 Radiography: Normal vs Viral Pneumonia).
Order inside this single run: (1) train the 'fix' model on primary data only, (2) only then open the external images,
(3) score the unfixed final model and the fixed model once each, (4) write results/external_lock.json. A second run is refused.
  python p2_external.py --confirm-once [--fix clahe|heq|strong_aug]"""
import argparse, json, datetime, hashlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from lib import *
from lib2 import *
from train import data, train_cfg
from nn_lib import predict

ap = argparse.ArgumentParser(); ap.add_argument("--confirm-once", action="store_true"); ap.add_argument("--fix", default="clahe", choices=["clahe", "heq", "strong_aug"])
ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--res", type=int, default=None); a = ap.parse_args()
LOCK = RESULTS / "external_lock.json"
if LOCK.exists(): raise SystemExit(f"REFUSED: external data was already used ({LOCK}). It may be touched exactly once.")
if not a.confirm_once: raise SystemExit("Add --confirm-once: you are about to spend the single permitted look at the external hospital.")

fi = final_info(); cfg = dict(fi["cfg"]); thr0 = json.load(open(RESULTS / "threshold.json"))["threshold"]
if a.epochs: cfg["epochs"] = a.epochs
if a.res: cfg["res"] = a.res
# ---- (1) fixed model: trained on primary data only
fixcfg = dict(cfg, prep={"clahe": "clahe", "heq": "heq"}.get(a.fix, "none"), aug="strong" if a.fix == "strong_aug" else cfg["aug"])
df, X, M = data(fixcfg["res"], fixcfg["prep"]); tr, va = (df.split == "train").values, (df.split == "val").values; y = df.label.values
model_fix, _, _ = train_cfg(fixcfg, fi["best_seed"], df, X, M, tr, va, verbose=False)
p_fix_va = predict(model_fix, X[va]); thr_fix, *_ = best_threshold(y[va], p_fix_va)
model_base, bcfg, _ = load_model(fi["name"], fi["best_seed"])
dfb, Xb, _ = load_split(bcfg["res"], bcfg.get("prep", "none")); te = (dfb.split == "test").values
internal_base = compute_metrics(dfb.label.values[te], predict(model_base, Xb[te]), thr0)
internal_fix = compute_metrics(y[te], predict(model_fix, X[te]), thr_fix)

# ---- (2)-(3) open the external data now
ext = index_external(); info_e = get_info(ext, "external"); ye = ext.label.values
Xe_base, _ = get_arrays(ext, "external", bcfg["res"], bcfg.get("prep", "none")); Xe_fix, _ = get_arrays(ext, "external", fixcfg["res"], fixcfg["prep"])
res = {}
for nm, mdl, Xe, thr, internal in (("final (unfixed)", model_base, Xe_base, thr0, internal_base), (f"fixed: {a.fix}", model_fix, Xe_fix, thr_fix, internal_fix)):
    pe = predict(mdl, Xe); m = compute_metrics(ye, pe, thr)
    res[nm] = dict(ext=m, internal=internal, thr=thr, cost_internal=float(expected_cost(internal["recall"], 1 - internal["specificity"])),
                   cost_external=float(expected_cost(m["recall"], 1 - m["specificity"])))
LOCK.write_text(json.dumps(dict(when=datetime.datetime.now().isoformat(), fix=a.fix, n_images=len(ext), script_sha1=hashlib.sha1(open(__file__, "rb").read()).hexdigest()), indent=1))
save_json(res, RESULTS / "external_results.json")

# ---- pixel statistics: primary vs external
prim = get_info(index_primary(), "primary")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); rows = []
for a_, f in zip(ax, ("mean_int", "std_int", "area")):
    a_.hist(prim[f], bins=40, density=True, alpha=.5, label="primary (Guangzhou, children)"); a_.hist(info_e[f], bins=40, density=True, alpha=.5, label="external"); a_.set_title(f); a_.legend(fontsize=7)
    ks = stats.ks_2samp(prim[f], info_e[f]); rows.append(dict(feature=f, primary_median=prim[f].median(), external_median=info_e[f].median(), ks_stat=ks.statistic, ks_p=ks.pvalue))
plt.tight_layout(); plt.savefig(FIGS / "p2_external_pixelstats.png", dpi=130); plt.close()
ps = pd.DataFrame(rows); ps.to_csv(TABLES / "p2_external_pixelstats.csv", index=False)
tab = pd.DataFrame([dict(model=k, threshold=v["thr"], **{f"internal_{m}": v["internal"][m] for m in ("recall", "specificity", "roc_auc")},
                         **{f"external_{m}": v["ext"][m] for m in ("recall", "specificity", "roc_auc", "pr_auc")},
                         cost_internal=v["cost_internal"], cost_external=v["cost_external"]) for k, v in res.items()])
tab.to_csv(TABLES / "p2_external.csv", index=False)
b, f_ = res["final (unfixed)"], list(res.values())[1]
helped = f_["ext"]["roc_auc"] > b["ext"]["roc_auc"] + 0.01 and f_["cost_external"] < b["cost_external"]
md = ["## Part 2.6 Generalisation test (external hospital, evaluated exactly once)\n",
      f"{len(ext)} external images (Normal / Viral Pneumonia folders only; pneumonia fraction {ye.mean():.2f}). Thresholds were fixed on the primary validation split *before* the external data was opened; nothing was tuned on it. "
      f"Lock file: `results/external_lock.json`.\n", md_table(tab), "",
      f"**Drop from internal test to external:** ROC-AUC {b['internal']['roc_auc']:.3f} → {b['ext']['roc_auc']:.3f}; at the camp threshold recall {b['internal']['recall']:.3f} → {b['ext']['recall']:.3f}, specificity {b['internal']['specificity']:.3f} → {b['ext']['specificity']:.3f}; "
      f"expected camp cost per patient {b['cost_internal']:.3f} → {b['cost_external']:.3f}.\n",
      "Pixel statistics, primary vs external (`figures/p2_external_pixelstats.png`; KS test):\n", md_table(ps, "{:.4g}"), "",
      "**Explaining the drop.** Distribution mismatch: the external images differ in intensity and contrast statistics (table above), in resolution (fixed-size PNG vs mixed-size JPEG), in population (adults and children vs children aged 1-5) and in label definition (viral pneumonia only). "
      "A model that learned acquisition cues from the primary hospital (task 4 and the metadata branch of task 8) loses exactly those cues; the threshold, chosen from the primary score distribution, no longer sits at the right place for the shifted score distribution.\n",
      f"**Fix tried (one only): `{a.fix}`** - " + ("it **helped** (higher external AUC and lower expected cost)." if helped else "it **did not help** measurably (external AUC / cost not clearly better); report it as a failed fix.") +
      " Honest caveat: a single external dataset and one training seed, so differences of about 0.01 AUC are not distinguishable from noise.\n"]
write_fragment("p2_external", "\n".join(md))
