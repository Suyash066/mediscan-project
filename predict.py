"""predict.py IMAGE_PATH  ->  prints the class and probability (and the zone / refusal reason).
Self-contained: needs only deploy/ (model + config + ood stats), torch, opencv, pillow, numpy."""
import sys, json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from models import build_model

DEPLOY = Path(__file__).resolve().parent / "deploy"
_STATE = {}


def _load():
    if _STATE: return _STATE
    ck = torch.load(DEPLOY / "arogyanet.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    z = np.load(DEPLOY / "ood.npz")
    _STATE.update(model=m, cfg=json.load(open(DEPLOY / "config.json")), mu=z["mu"], P=z["P"]); torch.set_num_threads(4)
    return _STATE


def assess(img: Image.Image) -> dict:
    s = _load(); c = s["cfg"]
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    spread = float((np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean() / 2)
    g = np.array(img.convert("L"), dtype=np.uint8)
    if spread > c["colour_spread_max"]:
        return dict(label="CANNOT ASSESS", prob=None, zone="reject", note=f"colour photograph (channel spread {spread:.1f}), not a chest X-ray")
    if g.std() / 255.0 < c["blank_std_min"]:
        return dict(label="CANNOT ASSESS", prob=None, zone="reject", note="blank / almost uniform image")
    if c["prep"] == "clahe": g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    elif c["prep"] == "heq": g = cv2.equalizeHist(g)
    x = torch.from_numpy(cv2.resize(g, (c["res"], c["res"]), interpolation=cv2.INTER_AREA)).float().div(255)[None, None]
    with torch.no_grad():
        e = s["model"].embed(x); logit = s["model"].fc(e).squeeze()
    d = float(np.sqrt((e.numpy()[0] - s["mu"]) @ s["P"] @ (e.numpy()[0] - s["mu"])))
    if d > c["ood_threshold"]:
        return dict(label="CANNOT ASSESS", prob=None, zone="reject", note=f"image does not resemble the chest X-rays the model was trained on (distance {d:.1f} > {c['ood_threshold']:.1f})")
    p = float(torch.sigmoid(logit))
    if p < c["threshold"]: zone, label = "normal", "NORMAL"
    elif p < c["t_high"]: zone, label = "low_confidence", "LOW CONFIDENCE - refer to a clinician"
    else: zone, label = "pneumonia", "PNEUMONIA suspected - refer to a clinician"
    return dict(label=label, prob=p, zone=zone, note=f"threshold {c['threshold']:.3f}, high {c['t_high']:.3f}")


def predict_path(path: str) -> dict:
    return assess(Image.open(path))


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: python predict.py IMAGE_PATH")
    r = predict_path(sys.argv[1])
    print(f"{r['label']}" + (f"  probability_of_pneumonia={r['prob']:.3f}" if r["prob"] is not None else "") + f"  ({r['note']})")
    print("Screening aid only - not a diagnosis.")
