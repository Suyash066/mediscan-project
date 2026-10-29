# MediScan Audit - reproducible repo (PyTorch)

## 0. Honest status
* All code runs end to end (`make smoke` = whole pipeline on a tiny **fake** dataset, ~2 min on CPU; numbers from it mean nothing).
* It has **not** been run on the real Kaggle data (no internet in the authoring sandbox). Run it yourself: every number, figure and table in the report then comes from your run.
* `reports/Report.md` / `Part1_audit.md` are generated with all tables/figures filled in by scripts; the interpretation sentences are marked `<!-- WRITE: ... -->` and must be written by you after you look at *your* results (11 slots). Do not submit them empty.
* Tested with Python 3.12, torch 2.14, scikit-learn 1.9, numpy 2.5. `requirements.txt` pins an older, conservative set (Colab-era) that was **not** tested here; `run_all.sh` saves the real `pip freeze` to `results/pip_freeze.txt`, which is the true record.

## 1. Data layout (not included)
```
data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg        # Kaggle "Chest X-Ray Images (Pneumonia)"
data/covid19_radiography/{Normal,Viral Pneumonia}/images/*.png    # COVID-19 Radiography Database (only these two folders are read)
```
Set another location with `MEDISCAN_DATA=/path`. All CSVs store paths relative to it.

## 2. Run
```
pip install -r requirements.txt
bash run_all.sh            # or: make all   (FIX=heq / TRIALS=20 are optional env vars)
python app.py              # Gradio demo (after run_all)      python predict.py some_xray.jpeg
```
Put a real dog photo at `tools/dog.jpg` *before* `run_all.sh` (or run `python tools/ood_tests.py --image dog.jpg` later) - none ships with the repo.
Then fill the `WRITE` slots and: `pandoc reports/Report.md -o reports/Report.pdf --resource-path=reports` (5-7 pages: trim tables if longer).
**Commit `reports/hypotheses.md` (edited to your own predictions) to git before running `train.py`.**

## 3. Where each task lives
| Task | Script -> output |
|---|---|
| P1.1 inventory, pixel stats | `audit.py` -> `reports/fragments/p1_audit.md`, `figures/p1_pixel_stats.png` |
| P1.2 duplicates / leakage | `audit.py` (pHash, worst-5 figure) |
| P1.3 val set CI + new split | `audit.py`, `split.py` -> `splits/split.csv` (group-aware, asserts no leakage) |
| P1.4 shortcut hunt, P1.5 baselines | `baselines.py` |
| P1.6 transfer, P1.7 ArogyaNet, P1.8 two-branch, P1.9 ablations | `models.py`, `train.py`, `aggregate.py`, `arch_diagram.py`, `reports/hypotheses.md` |
| P2.1 imbalance vs threshold, cost | `p2_threshold.py` |
| P2.2 evaluation + 5-fold CV | `p2_eval.py` |
| P2.3 tuning, over-fitting curves | `p2_tune.py` |
| P2.4 ensembles | `p2_ensemble.py` |
| P2.5 feature space | `p2_embed.py` |
| P2.6 external test (once) | `p2_external.py --confirm-once` (writes a lock file, refuses a 2nd run) |
| P2.7 ship + CEO | `export.py`, `predict.py`, `app.py`, `tools/ood_tests.py`, `ceo_arithmetic.py` |
| P2.8 report | `make_report.py` -> `reports/Report.md`, `Part1_audit.md`, `CEO_report.md` |

## 4. Design choices to be able to defend
* Split is **group-aware** (pHash near-duplicate clusters + patient ids parsed from file names), so duplicates cannot straddle train/test.
* ArogyaNet (~130k params): depthwise-separable blocks, **D1 residual skips**, **D2 squeeze-excitation**, global-average-pooling head; both D1 and D2 are ablated over 3 seeds with paired differences and an explicit "inside the noise" verdict.
* The threshold is chosen on **validation** by expected cost at 10% prevalence (miss = 20, false alarm = 1); TPR/FPR are prevalence-free, so re-weighting is valid.
* Augmentation has **no vertical flip** (see report section on task 9).
* The external dataset is adults+children, viral-only labels: expect a drop; report it, explain it, try one fix.
