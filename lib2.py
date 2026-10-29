"""Helpers for Part 2: load checkpoints / predictions, cost-optimal threshold."""
import json
import numpy as np, torch
from sklearn.metrics import roc_curve
from lib import *
from models import build_model
from nn_lib import predict, meta_norm


def load_model(name, seed):
    ck = torch.load(MODELS / f"{name}_s{seed}.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    st = (np.array(ck["meta_stats"][0], np.float32), np.array(ck["meta_stats"][1], np.float32)) if ck["meta_stats"] else None
    return m, ck["cfg"], st


def model_probs(name, seed):
    """Probabilities of a trained model on ALL rows of splits/split.csv (aligned with load_split order)."""
    m, cfg, st = load_model(name, seed)
    df, X, M = load_split(cfg["res"], cfg.get("prep", "none"))
    return df, predict(m, X, meta_norm(M, st) if st else None), cfg


def stored_probs(name, seed):
    z = np.load(RESULTS / "preds" / f"{name}_s{seed}.npz", allow_pickle=True)
    return z["p"], z["split"], z["y"]


def final_info():
    f = RESULTS / "final_config.json"
    if not f.exists(): raise SystemExit("run p2_tune.py first (it writes results/final_config.json)")
    return json.load(open(f))


def cost_curve(y, p, prevalence=CAMP_PREVALENCE):
    """TPR/FPR measured on (y,p) re-weighted to `prevalence`; returns thr, tpr, fpr, cost (per patient)."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.where(np.isinf(thr), 1.0 + 1e-6, thr)
    return thr, tpr, fpr, expected_cost(tpr, fpr, prevalence)


def best_threshold(y, p, prevalence=CAMP_PREVALENCE):
    thr, tpr, fpr, cost = cost_curve(y, p, prevalence)
    m = cost.min(); k = np.where(cost <= m + 1e-12)[0][0]    # roc_curve thresholds descend -> first = highest threshold among ties
    return float(min(thr[k], 1.0)), float(m), float(tpr[k]), float(fpr[k])
