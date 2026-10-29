"""Training / inference helpers shared by all neural-network scripts."""
import copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from lib import set_seed

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment(x, policy, gen=None):
    """GPU batch augmentation. policy: none | std (hflip + rot +-10deg + zoom 0.9-1.1 + shift 5%) | strong (std + brightness/contrast).
    NO vertical flip: an upside-down chest never occurs and puts stomach bubble / diaphragm / heart on the wrong side of the image."""
    if policy == "none":
        return x
    B, dev = x.shape[0], x.device
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
    ang = r(-10, 10) * np.pi / 180; zoom = r(0.9, 1.1); tx, ty = r(-.05, .05), r(-.05, .05)
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    c, s = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    theta = torch.stack([torch.stack([c * flip, -s, tx], 1), torch.stack([s * flip, c, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    if policy == "strong":
        con, bri = r(0.6, 1.4).view(B, 1, 1, 1), r(-0.2, 0.2).view(B, 1, 1, 1)
        x = ((x - 0.5) * con + 0.5 + bri).clamp(0, 1)
    return x


def meta_stats(meta):
    m = np.c_[meta[:, 0], meta[:, 1], np.log(meta[:, 2])].astype(np.float32)
    return m.mean(0), m.std(0) + 1e-6

def meta_norm(meta, stats):
    m = np.c_[meta[:, 0], meta[:, 1], np.log(meta[:, 2])].astype(np.float32)
    return ((m - stats[0]) / stats[1]).astype(np.float32)


@torch.no_grad()
def predict(model, X, M=None, bs=128, embed=False):
    dev = device(); model.eval().to(dev); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
        mb = torch.from_numpy(M[i:i + bs]).to(dev) if M is not None else None
        o = model.embed(xb, mb) if embed else torch.sigmoid(model(xb, mb))
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)


def _eval_loss(model, X, M, y, bs=128):
    p = predict(model, X, M, bs)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return loss, auc, p


def fit(model, Xtr, Mtr, ytr, Xva, Mva, yva, *, epochs=25, lr=2e-3, wd=1e-4, bs=32, aug="std", pos_weight=None,
        patience=6, early_stop=True, seed=0, verbose=True):
    """Train with AdamW + cosine schedule. Early stopping monitors validation BCE; the best-epoch weights are restored
    when early_stop=True (otherwise the last epoch is kept - used for the 'unregularised' curve).
    Returns (model, history dict)."""
    set_seed(seed, log=False)
    dev = device(); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.tensor(pos_weight, device=dev, dtype=torch.float32) if pos_weight else None
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    gen_dev = torch.Generator(device=dev); gen_dev.manual_seed(seed)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.float32))
    Mt = torch.from_numpy(Mtr) if Mtr is not None else None
    hist = dict(train_loss=[], val_loss=[], val_auc=[], lr=[]); best = (1e9, None, -1); bad = 0
    for ep in range(epochs):
        model.train(); idx = torch.randperm(len(Xt), generator=g); tot = 0.0; n = 0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            if len(b) < 2: continue                                        # BatchNorm needs >1 sample
            xb = Xt[b].to(dev).float().div(255).unsqueeze(1); yb = yt[b].to(dev)
            mb = Mt[b].to(dev) if Mt is not None else None
            xb = augment(xb, aug, gen_dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logit = model(xb, mb)
                loss = F.binary_cross_entropy_with_logits(logit.float(), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(b); n += len(b)
        sched.step()
        vl, va, _ = _eval_loss(model, Xva, Mva, yva)
        hist["train_loss"].append(tot / n); hist["val_loss"].append(vl); hist["val_auc"].append(va); hist["lr"].append(opt.param_groups[0]["lr"])
        if verbose: print(f"  ep{ep+1:02d} train {tot/n:.4f} val {vl:.4f} auc {va:.4f}", flush=True)
        if vl < best[0] - 1e-4:
            best = (vl, copy.deepcopy(model.state_dict()), ep); bad = 0
        else:
            bad += 1
            if early_stop and bad >= patience:
                break
    hist["best_epoch"] = best[2] + 1; hist["epochs_run"] = len(hist["val_loss"]); hist["best_val_loss"] = best[0]
    if early_stop and best[1] is not None:
        model.load_state_dict(best[1])
    return model, hist


def latency_ms(model, res, n=50, meta_dim=0):
    """Median/p95 single-image CPU latency in ms (batch 1, 4 threads)."""
    torch.set_num_threads(4)
    m = model.to("cpu").eval(); x = torch.rand(1, 1, res, res); mt = torch.zeros(1, 3) if meta_dim else None
    ts = []
    with torch.no_grad():
        for _ in range(5): m(x, mt)
        for _ in range(n):
            t = time.perf_counter(); m(x, mt); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))
