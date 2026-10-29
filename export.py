"""Part 2, task 7a: package the final model for the camp laptop -> deploy/arogyanet.pt, deploy/config.json, deploy/ood.npz.
Defines the 3-zone decision bands and the out-of-distribution guard, measures file size and CPU latency.
Usage: python export.py"""
import json, os, time
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve
from lib import *
from lib2 import *
from nn_lib import predict, latency_ms

DEPLOY = ROOT / "deploy"; DEPLOY.mkdir(exist_ok=True)
fi = final_info(); th = json.load(open(RESULTS / "threshold.json")); thr = th["threshold"]
m, cfg, st = load_model(fi["name"], fi["best_seed"]); assert not cfg.get("meta"), "deployed model must not use the metadata branch"
df, X, M = load_split(cfg["res"], cfg.get("prep", "none")); y = df.label.values; sp = df.split.values
va = sp == "val"; pv = predict(m, X[va])
f, t, thr_r = roc_curve(y[va], pv); thr_r = np.where(np.isinf(thr_r), 1.0, thr_r)
ppv = ppv_at_prevalence(t, f)
ok = np.where((thr_r >= thr) & (ppv >= 0.5))[0]
t_high = float(thr_r[ok].min()) if len(ok) else 1.0       # smallest score where >=50% of alarms are true at camp prevalence
# OOD guard: Mahalanobis distance of the embedding to the training-set embedding cloud
E = predict(m, X, None, embed=True); Etr = E[sp == "train"]
lw = LedoitWolf().fit(Etr); mu, P = Etr.mean(0), lw.precision_
maha = lambda e: np.sqrt(np.einsum("ij,jk,ik->i", e - mu, P, e - mu))
dtr, dva, dte = maha(Etr), maha(E[va]), maha(E[sp == 'test'])
ood_thr = float(1.25 * max(np.percentile(dtr, 99.5), np.percentile(dva, 99.5)))      # 25% head-room: train embeddings are tighter than unseen ones
false_reject = float((dte > ood_thr).mean())
np.savez(DEPLOY / "ood.npz", mu=mu, P=P)
torch.save(dict(state=m.state_dict(), cfg=cfg), DEPLOY / "arogyanet.pt")
conf = dict(res=cfg["res"], prep=cfg.get("prep", "none"), threshold=thr, t_high=t_high, ood_threshold=ood_thr, camp_prevalence=CAMP_PREVALENCE,
            colour_spread_max=12.0, blank_std_min=0.03, model=f"{fi['name']}_s{fi['best_seed']}")
json.dump(conf, open(DEPLOY / "config.json", "w"), indent=1)
size_mb = os.path.getsize(DEPLOY / "arogyanet.pt") / 1e6
lat = latency_ms(m, cfg["res"])
from predict import predict_path
sample = ROOT / "data" / df.path[np.where(sp == "test")[0][0]] if not (DATA / df.path[0]).exists() else DATA / df.path[np.where(sp == "test")[0][0]]
ts = []
for _ in range(20):
    t0 = time.perf_counter(); predict_path(str(sample)); ts.append((time.perf_counter() - t0) * 1e3)
e2e = (float(np.median(ts)), float(np.percentile(ts, 95)))
save_json(dict(size_mb=size_mb, model_latency_ms=lat, end_to_end_ms=e2e, **conf), RESULTS / "deploy_check.json")
md = ["## Part 2.7 Shipping\n", f"* Model file `deploy/arogyanet.pt`: **{size_mb:.2f} MB** (limit 25 MB) - {'OK' if size_mb < 25 else 'TOO BIG'}.",
      f"* CPU latency, batch 1, 4 threads: model forward **{lat[0]:.1f} ms median / {lat[1]:.1f} ms p95**; end to end (read JPEG + resize + forward + guards) **{e2e[0]:.1f} / {e2e[1]:.1f} ms** - limit 200 ms: {'OK' if e2e[1] < 200 else 'TOO SLOW'}. (Measured on *this* machine; repeat on the camp laptop.)",
      f"* Decision zones on the output probability p: p < {thr:.3f} -> NORMAL; {thr:.3f} <= p < {t_high:.3f} -> **LOW CONFIDENCE - refer to a clinician**; p >= {t_high:.3f} -> PNEUMONIA suspected, refer. "
      f"{t_high:.3f} is the smallest score at which at least half of the alarms are real at {CAMP_PREVALENCE:.0%} prevalence (validation estimate).",
      f"* Not-an-X-ray guards (run before the network): colour photographs (channel spread > 12/255), blank images (grey std < 0.03), and embeddings farther than {ood_thr:.2f} (Mahalanobis, 1.25 x the 99.5th percentile of train/val) from the training cloud are rejected with 'cannot assess'. False-reject rate on held-out test X-rays: {100*false_reject:.1f}% (reported, not tuned).\n"]
write_fragment("p2_ship", "\n".join(md))
