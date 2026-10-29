"""Part 1, tasks 6-9 (and the model zoo reused by Part 2).
  python train.py --exps transfer_resnet18 arogya arogya_noskip arogya_nose arogya_meta arogya_res96 arogya_augnone --seeds 0 1 2
Each (experiment, seed) writes results/part1/<exp>_s<seed>.json, results/preds/<exp>_s<seed>.npz and models/<exp>_s<seed>.pt.
The test split is scored and stored but only read by aggregate.py (never used to choose anything)."""
import argparse, json, functools
import numpy as np, pandas as pd, torch
from lib import *
from models import build_model, n_params, arch_table
from nn_lib import fit, predict, meta_stats, meta_norm, latency_ms, device

BASE = dict(arch="arogya", res=224, aug="std", use_skip=True, use_se=True, meta=False, dropout=0.2,
            lr=2e-3, wd=1e-4, bs=32, epochs=25, patience=6, cw=False, prep="none")
EXPS = {
    "transfer_resnet18": dict(BASE, arch="resnet18", lr=3e-4, epochs=10, patience=4),
    "transfer_mobilenet_v3_small": dict(BASE, arch="mobilenet_v3_small", lr=5e-4, epochs=10, patience=4),
    "transfer_efficientnet_b0": dict(BASE, arch="efficientnet_b0", lr=5e-4, epochs=10, patience=4),
    "arogya": dict(BASE),
    "arogya_noskip": dict(BASE, use_skip=False),          # ablation of design decision D1
    "arogya_nose": dict(BASE, use_se=False),              # ablation of design decision D2
    "arogya_meta": dict(BASE, meta=True),                 # two-branch experiment (task 8)
    "arogya_res96": dict(BASE, res=96),                   # resolution ablation (task 9)
    "arogya_augnone": dict(BASE, aug="none"),             # augmentation ablation (task 9)
    "arogya_cw": dict(BASE, cw=True),                     # Part 2: class-weighted loss
}

@functools.lru_cache(maxsize=4)
def data(res, prep):
    return load_split(res, prep)


def train_cfg(cfg, seed, df, X, M, tr, va, verbose=True, **over):
    """Train one model on rows `tr`, early-stop on rows `va`. Returns model, hist, meta stats."""
    cfg = dict(cfg, **over)
    set_seed(seed, log=True)
    model = build_model(cfg)
    y = df.label.values
    st = meta_stats(M[tr]) if cfg.get("meta") else None
    Mn = meta_norm(M, st) if st else None
    pw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)) if cfg.get("cw") else None
    model, hist = fit(model, X[tr], Mn[tr] if st else None, y[tr], X[va], Mn[va] if st else None, y[va],
                      epochs=cfg["epochs"], lr=cfg["lr"], wd=cfg["wd"], bs=cfg["bs"], aug=cfg["aug"], pos_weight=pw,
                      patience=cfg["patience"], early_stop=cfg.get("early_stop", True), seed=seed, verbose=verbose)
    return model, hist, st


def run(name, cfg, seed, force=False):
    out = RESULTS / "part1" / f"{name}_s{seed}.json"
    if out.exists() and not force:
        print("skip", out.name); return
    df, X, M = data(cfg["res"], cfg["prep"])
    tr, va, te = [(df.split == s).values for s in ("train", "val", "test")]
    print(f"== {name} seed {seed}  train {tr.sum()} val {va.sum()} test {te.sum()}  device {device()}", flush=True)
    model, hist, st = train_cfg(cfg, seed, df, X, M, tr, va)
    Mn = meta_norm(M, st) if st else None
    p = predict(model, X, Mn)
    y = df.label.values
    res = dict(exp=name, seed=seed, cfg=cfg, n_params=n_params(model), best_epoch=hist["best_epoch"], history=hist,
               val=compute_metrics(y[va], p[va]), test=compute_metrics(y[te], p[te]), meta_stats=None if st is None else [st[0].tolist(), st[1].tolist()])
    if seed == SEEDS[0]:
        res["latency_ms_cpu"] = latency_ms(model, cfg["res"], meta_dim=3 if cfg["meta"] else 0)
        if name == "arogya":
            tab = arch_table(build_model(cfg), cfg["res"]); tab.to_csv(TABLES / "arogyanet_architecture.csv", index=False)
    save_json(res, out)
    (RESULTS / "preds").mkdir(exist_ok=True)
    np.savez(RESULTS / "preds" / f"{name}_s{seed}.npz", p=p, split=df.split.values, y=y)
    torch.save(dict(state=model.state_dict(), cfg=cfg, meta_stats=res["meta_stats"]), MODELS / f"{name}_s{seed}.pt")
    print(f"   val bal_acc {res['val']['bal_acc']:.3f} auc {res['val']['roc_auc']:.3f} | params {res['n_params']:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=list(EXPS)[:9])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--res", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); log_env()
    for e in a.exps:
        cfg = dict(EXPS[e])
        if a.epochs: cfg["epochs"] = a.epochs
        if a.res: cfg["res"] = a.res
        for s in a.seeds:
            run(e, cfg, s, a.force)
