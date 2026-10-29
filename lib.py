"""Shared, torch-free utilities for the MediScan audit (paths, seeds, data index,
image cache, perceptual hashing, metrics, markdown helpers).

Every script imports from here so that seeds, paths and constants are defined once.
"""
from __future__ import annotations
import os, re, io, json, random, hashlib, platform, datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MEDISCAN_DATA", ROOT / "data"))
EXTERNAL = DATA / "covid19_radiography"
CACHE, RESULTS, SPLITS, MODELS = ROOT / "cache", ROOT / "results", ROOT / "splits", ROOT / "models"
FIGS, FRAGS = ROOT / "reports" / "figures", ROOT / "reports" / "fragments"
TABLES = RESULTS / "tables"
for _d in (CACHE, RESULTS, SPLITS, MODELS, FIGS, FRAGS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpeg", ".jpg", ".png"}
SEEDS = [0, 1, 2]                 # the three seeds used everywhere
GLOBAL_SEED = 1234                # for splits / non-model randomness
NEAR_DUP_T = 4                    # pHash Hamming distance (of 64 bits) called "near-duplicate"
CAMP_PREVALENCE = 0.10            # pneumonia fraction at the camps (about 9:1 NORMAL)
COST_FN, COST_FP = 20.0, 1.0      # a missed pneumonia costs ~20x a false alarm


# ----------------------------------------------------------------------------- seeds
def set_seed(seed: int, log: bool = True):
    """Seed Python, NumPy and (if installed) PyTorch; log the seed to results/seeds.log."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    if log:
        with open(RESULTS / "seeds.log", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} seed={seed} "
                    f"script={os.path.basename(os.sys.argv[0])}\n")


def log_env():
    with open(RESULTS / "env.json", "w") as f:
        info = dict(python=platform.python_version(), platform=platform.platform())
        for m in ("numpy", "pandas", "sklearn", "cv2", "PIL", "torch", "torchvision", "skimage"):
            try:
                info[m] = __import__(m).__version__
            except Exception:
                pass
        json.dump(info, f, indent=1)


# ----------------------------------------------------------------------------- data index
def _find_primary_root() -> Path:
    for cand in (DATA / "chest_xray", DATA / "chest_xray" / "chest_xray", DATA):
        if (cand / "train" / "NORMAL").exists():
            return cand
    raise FileNotFoundError(f"Could not find train/NORMAL under {DATA}. See README (data layout).")


def index_primary() -> pd.DataFrame:
    """One row per image of Rohit's download, exactly as shipped (orig_split = his split)."""
    root = _find_primary_root()
    rows = []
    for split in ("train", "val", "test"):
        for cls, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
            for p in sorted((root / split / cls).glob("*")):
                if p.suffix.lower() in IMG_EXT and not p.name.startswith("."):
                    rows.append(dict(path=p.relative_to(DATA).as_posix(), fname=p.name, orig_split=split, label=lab))
    df = pd.DataFrame(rows)
    return df


def index_external() -> pd.DataFrame:
    """External hospital (COVID-19 Radiography Database): Normal=0, Viral Pneumonia=1 ONLY."""
    rows = []
    for folder, lab in (("Normal", 0), ("Viral Pneumonia", 1)):
        base = None
        for cand in (EXTERNAL, EXTERNAL / "COVID-19_Radiography_Dataset"):
            if (cand / folder).exists():
                base = cand / folder
        if base is None:
            raise FileNotFoundError(f"External folder '{folder}' not found under {EXTERNAL}")
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append(dict(path=p.relative_to(DATA).as_posix(), fname=p.name, orig_split="external", label=lab))
    return pd.DataFrame(rows)


def patient_group(fname: str, label: int) -> str:
    """Heuristic patient id from the Kaggle file names (the data ships without patient ids).
    PNEUMONIA: person123_bacteria_456.jpeg -> P123 ; NORMAL: IM-0115-0001.jpeg -> N0115,
    NORMAL2-IM-0373-0001-0002.jpeg -> M0373. Anything else is its own group."""
    m = re.match(r"person(\d+)_", fname)
    if m: return f"P{m.group(1)}"
    m = re.match(r"NORMAL2-IM-(\d+)", fname)
    if m: return f"M{m.group(1)}"
    m = re.match(r"IM-(\d+)", fname)
    if m: return f"N{m.group(1)}"
    return f"F:{fname}"


# ----------------------------------------------------------------------------- image I/O
def abspath(p) -> Path:
    """Paths in all CSVs are relative to MEDISCAN_DATA so results are machine independent."""
    p = Path(p)
    return p if p.is_absolute() else DATA / p


def load_gray(path) -> np.ndarray:
    from PIL import Image
    with Image.open(abspath(path)) as im:
        return np.array(im.convert("L"), dtype=np.uint8)


def phash64(g: np.ndarray) -> int:
    import cv2
    im = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(im)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    return _LUT[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1).astype(np.uint8)


def near_dup_pairs(hashes: np.ndarray, thr: int, chunk: int = 256):
    """All pairs (i<j) with Hamming(pHash_i, pHash_j) <= thr -> arrays i, j, d."""
    h = hashes.astype(np.uint64)
    I, J, D = [], [], []
    for i0 in range(0, len(h), chunk):
        x = h[i0:i0 + chunk, None] ^ h[None, :]
        d = popcount64(x)
        ii, jj = np.nonzero(d <= thr)
        keep = jj > (ii + i0)
        I.append(ii[keep] + i0); J.append(jj[keep]); D.append(d[ii[keep], jj[keep]])
    if not I:
        return np.array([], int), np.array([], int), np.array([], int)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def get_info(df: pd.DataFrame, name: str = "primary") -> pd.DataFrame:
    """Per-image facts computed once and cached: size, mode, true-grey flag, intensity
    statistics, sha1 of the file, 64-bit pHash. Row order == df order."""
    from PIL import Image
    f = CACHE / f"info_{name}.csv"
    if f.exists():
        info = pd.read_csv(f, dtype={"phash": str})
        if len(info) == len(df) and (info["path"].values == df["path"].values).all():
            return info
    rows = []
    for p in df["path"]:
        with open(abspath(p), "rb") as fh:
            raw = fh.read()
        im = Image.open(io.BytesIO(raw))
        w, h, mode = im.size[0], im.size[1], im.mode
        arr = np.array(im)
        true_gray = (arr.ndim == 2) or (arr.ndim == 3 and np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]))
        g = np.array(im.convert("L"), dtype=np.uint8)
        rows.append(dict(path=p, w=w, h=h, mode=mode, true_gray=bool(true_gray),
                         mean_int=g.mean() / 255.0, std_int=g.std() / 255.0,
                         sha1=hashlib.sha1(raw).hexdigest(), phash=format(phash64(g), "016x")))
    info = pd.DataFrame(rows)
    info["aspect"] = info["w"] / info["h"]
    info["area"] = info["w"] * info["h"]
    info.to_csv(f, index=False)
    return info


def hashes_of(info: pd.DataFrame) -> np.ndarray:
    return np.array([int(x, 16) for x in info["phash"]], dtype=np.uint64)


def prep_image(g: np.ndarray, prep: str) -> np.ndarray:
    import cv2
    if prep == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if prep == "heq":
        return cv2.equalizeHist(g)
    return g


def get_arrays(df: pd.DataFrame, name: str, res: int, prep: str = "none"):
    """uint8 array (N,res,res) + metadata (N,3)=[mean intensity, aspect ratio, area] cached on disk."""
    import cv2
    f = CACHE / f"arr_{name}_{res}_{prep}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if len(z["X"]) == len(df) and (z["paths"] == df["path"].values).all():
            return z["X"], z["meta"]
    X = np.zeros((len(df), res, res), np.uint8)
    meta = np.zeros((len(df), 3), np.float32)
    for k, p in enumerate(df["path"]):
        g = load_gray(p)
        meta[k] = [g.mean() / 255.0, g.shape[1] / g.shape[0], g.shape[0] * g.shape[1]]
        X[k] = cv2.resize(prep_image(g, prep), (res, res), interpolation=cv2.INTER_AREA)
    np.savez_compressed(f, X=X, meta=meta, paths=df["path"].values)
    return X, meta


def load_split(res: int | None = None, prep: str = "none"):
    """Returns df (with split column from splits/split.csv) and optionally arrays."""
    df = pd.read_csv(SPLITS / "split.csv")
    if res is None:
        return df
    X, meta = get_arrays(df, "primary", res, prep)
    return df, X, meta


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y, p, thr: float = 0.5) -> dict:
    from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score)
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    rec, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec = div(tp, tp + fp)
    out = dict(thr=float(thr), acc=div(tp + tn, len(y)), bal_acc=np.nanmean([rec, spec]),
               precision=prec, recall=rec, specificity=spec,
               f1=div(2 * tp, 2 * tp + fp + fn), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def expected_cost(tpr, fpr, prevalence=CAMP_PREVALENCE, c_fn=COST_FN, c_fp=COST_FP):
    """Expected cost per patient: prev*c_fn*(1-TPR) + (1-prev)*c_fp*FPR (arrays ok)."""
    return prevalence * c_fn * (1 - np.asarray(tpr)) + (1 - prevalence) * c_fp * np.asarray(fpr)


def ppv_at_prevalence(tpr, fpr, prevalence=CAMP_PREVALENCE):
    tpr, fpr = np.asarray(tpr, float), np.asarray(fpr, float)
    den = prevalence * tpr + (1 - prevalence) * fpr
    return np.where(den > 0, prevalence * tpr / np.where(den > 0, den, 1), np.nan)


def bootstrap_ci(y, p, fn, n=300, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        v.append(fn(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ----------------------------------------------------------------------------- reporting
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def write_fragment(name: str, text: str):
    (FRAGS / f"{name}.md").write_text(text)
    print(f"[fragment] reports/fragments/{name}.md")


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
