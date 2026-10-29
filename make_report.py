"""Stitches every reports/fragments/*.md (all numbers produced by scripts) into
  reports/Part1_audit.md   (Part 1 deliverable)
  reports/Report.md        (the 5-7 page technical report: method + results + figures)
  reports/CEO_report.md    (the one-page memo to the CEO)
Text between <!-- WRITE: ... --> markers is the interpretation only YOU can write after looking at your numbers.
Usage: python make_report.py   (then e.g.  pandoc reports/Report.md -o reports/Report.pdf --resource-path=reports)"""
import json, re
from lib import *

def frag(n):
    f = FRAGS / f"{n}.md"
    return f.read_text() if f.exists() else f"*(fragment `{n}` missing: run the script that creates it)*\n"
def hyp():
    f = ROOT / "reports" / "hypotheses.md"; return f.read_text() if f.exists() else ""
def J(p):
    return json.load(open(p)) if Path(p).exists() else {}

P1_INTRO = """# Part 1 - Audit the data, then earn the model

Reproduce: `bash run_all.sh` (or `make part1`). Seeds are fixed (`GLOBAL_SEED=1234` for splits, model seeds 0/1/2) and logged in `results/seeds.log`; library versions in `results/env.json`.
All numbers below were written by the scripts named in each section (`audit.py`, `split.py`, `baselines.py`, `train.py`, `aggregate.py`).

"""
P1_WRITE_AUDIT = """
<!-- WRITE (task 1): in 3-4 sentences say what the class differences in brightness/size imply for a model trained on this data
     (use the AUC and Cohen's d numbers above; e.g. "a classifier can separate classes by X alone, so ..."). -->

<!-- WRITE (task 2): one sentence on what the cross-train/test near-duplicates do to the 98.4% (inflation, direction) and whether the worst-five figure shows the same child. -->
"""
P1_WRITE_SHORTCUT = """
<!-- WRITE (task 4): what the crippled model's score means - which cue is it using (see the metadata-only row) and why the real network can use it too. -->
<!-- WRITE (task 5): which baseline is the bar, and by how much (in balanced accuracy / PR-AUC with the CI) the deep model has to beat it. -->
"""
P1_WRITE_NN = """
### Transfer-learning expectation (write BEFORE looking at the table; keep it honest)
<!-- WRITE (task 6): what you expected to transfer from ImageNet RGB to grey radiographs (e.g. low-level edge/texture filters yes, object-level features no) and whether the ResNet18 numbers agree. -->

### Hypotheses (submitted before results; git-timestamped)
""" 
P1_WRITE_ABL = """
<!-- WRITE (task 7c): for D1 and D2 say whether the hypothesis held. If the verdict column says INSIDE THE NOISE, say so plainly - a failed hypothesis reported honestly scores full marks. -->

<!-- WRITE (task 8): if the metadata branch helped, what has the model learned about these patients (acquisition protocol / site, not pathology)? Would it hold at the camp? Link to the shortcut table (metadata_only row) and to the external result. -->

### Task 9: the medically wrong augmentation
**Vertical (up-down) flip.** A chest X-ray is acquired with a fixed orientation: diaphragm and stomach bubble at the bottom, heart apex on the patient's left, so an upside-down film is not a physiologically possible image; it forces the network to learn features for impossible anatomy and wastes capacity.
For *this* label it also moves cues that radiologists use (lower-zone consolidation, costophrenic angle blunting, retrocardiac opacity) to the wrong place, so the same pixels would mean something different, and the label is no longer consistent with the anatomy. 
A **horizontal flip** is the borderline case: it creates apparent dextrocardia/situs inversus and mirrors laterality markers; the pneumonia label itself usually survives, but the image is no longer a plausible patient, so this repo's `std` policy uses it only because the assignment lists flips, and the vertical flip is excluded (see `nn_lib.augment`). State in your write-up which of the two you consider wrong and why.
<!-- WRITE (task 9): interpret the resolution (96 vs 224) and augmentation (none vs std) rows in the paired-difference table. -->
"""

P2_INTRO = """
# Part 2 - Make it honest, then ship it

"""
def part1():
    return (P1_INTRO + frag("p1_audit") + P1_WRITE_AUDIT + "\n" + frag("p1_split") + "\n" + frag("p1_baselines") + P1_WRITE_SHORTCUT + "\n" + P1_WRITE_NN + hyp() + "\n"
            + "![diagram](figures/arogyanet_diagram.png)\n\n" + frag("p1_nn") + "\n![curves](figures/p1_arogya_curves.png)\n" + P1_WRITE_ABL)

CEO = """# Memo to the CEO - MediScan chest X-ray screening

**1. What Rohit's 98.4% really was.** Accuracy on a test folder that is 62.5% pneumonia and shares near-duplicate images (and patients) with the training folder, from one hospital, with no confidence interval. A model that says "pneumonia" for everyone scores 62.5% on that folder. The pilot (7 of 10 healthy children flagged, specificity about 30%) is what the number hid.

**2. What the rebuilt model achieves.** See the table below (numbers from `results/` via `ceo_arithmetic.py`, `p2_threshold.py`, `p2_external.py`). Operating point = the threshold that minimises expected cost at 10% prevalence with a missed pneumonia costing 20 false alarms.
<!-- WRITE: paste internal test sensitivity/specificity, the cross-validated mean ± std, and the external-hospital numbers (the drop) in plain language. -->

**3. What it must not be used for.** It is a screening aid that tells a nurse whom to refer, not a diagnosis; not for adults or other age groups, not for other hospitals/scanners until validated there (external drop above), not for other diseases, not on photographs of films, and never as the only reason to *clear* a child with symptoms. Images that are not chest X-rays are refused by the guard (tested in `tools/ood_tests.py`), but the guard is heuristic.

**4. Why "99% accuracy" is the wrong target.** Of the three instructions, this is the unreasonable one. At camp prevalence (about 9 healthy : 1 sick) doing nothing scores about 90%. 99% accuracy can be reached by missing 1 sick child in 10, and a model that catches 99% of the sick children with some false alarms scores only about 95% accuracy yet costs a third as much (table below). Ask for: *sensitivity at least X% (lower confidence bound) at specificity at least Y%, measured on an external site at camp prevalence*, plus the share of children referred.

**5. The other two instructions are reasonable and met:** the model file is far below 25 MB and runs in far under 200 ms per image on CPU (see the shipping numbers below; re-measure on the camp laptop).

"""
def ceo():
    return CEO + frag("p2_ceo") + "\n" + frag("p2_ship") + "\n" + frag("p2_ood")

def report():
    ex = J(RESULTS / "external_lock.json")
    h = """# The MediScan Audit - Technical report
*Every table and figure is produced by a script in this repository (`run_all.sh`); file names are given in each section.*

## 0. Summary and approach
**Problem.** A ResNet50 trained by a departed intern reported 98.4% accuracy but flagged 7 of 10 healthy children in a pilot. We (i) audit the data to find out why, (ii) rebuild with a leakage-free split, honest metrics and a deliberately small network (ArogyaNet, <2M parameters, no pretrained weights), (iii) choose an operating point by expected clinical cost at the camp prevalence, (iv) test once on an unseen hospital, and (v) ship a CPU model with a refusal guard.
<!-- WRITE: 4-6 sentence summary of YOUR headline findings (audit finding, best model + numbers with spread, chosen operating point, external drop, verdict on the CEO's three instructions). -->

**Pipeline.** `audit.py` -> `split.py` -> `baselines.py` -> `train.py` -> `aggregate.py` -> `p2_tune.py` -> `train.py --exps arogya_cw` -> `p2_threshold.py` -> `p2_eval.py` -> `p2_ensemble.py` -> `p2_embed.py` -> `p2_external.py --confirm-once` -> `export.py` -> `tools/ood_tests.py` -> `ceo_arithmetic.py` -> `make_report.py`.
Framework: PyTorch only. Seeds 0/1/2 for models, 1234 for splits. Test split touched only by `aggregate.py` and the Part-2 evaluation scripts; external data touched once (lock file `results/external_lock.json`: """ + (ex.get("when", "NOT YET RUN") if ex else "NOT YET RUN") + """).

"""
    return (h + part1().replace("# Part 1 - Audit the data, then earn the model", "# 1. Part 1 - Audit the data, then earn the model") + "\n" + P2_INTRO.replace("# Part 2", "# 2. Part 2") +
            frag("p2_threshold") + "\n" + frag("p2_eval") + "\n" + frag("p2_tune") + "\n" + frag("p2_ensemble") + "\n" + frag("p2_embed") + "\n" + frag("p2_external") +
            "\n" + frag("p2_ship") + "\n" + frag("p2_ood") + "\n# 3. Memo to the CEO\n" + ceo().replace("# Memo to the CEO - MediScan chest X-ray screening", "") +
            "\n# 4. Limitations\n<!-- WRITE: single-site training data; no patient ids (grouping is heuristic); 3 seeds only; labels are the dataset's own (not radiologist-verified for this use); external set is a different population and label definition; threshold chosen on a small validation slice; guard is heuristic. -->\n"
            "\n# 5. Reproducibility checklist\n`requirements.txt` pinned; `bash run_all.sh` regenerates every figure/table; seeds in `results/seeds.log`; environment in `results/env.json`; dataset paths via `MEDISCAN_DATA`.\n")

(ROOT / "reports" / "Part1_audit.md").write_text(part1())
(ROOT / "reports" / "Report.md").write_text(report())
(ROOT / "reports" / "CEO_report.md").write_text(ceo())
left = sum(len(re.findall(r"<!-- WRITE", (ROOT / "reports" / f).read_text())) for f in ("Report.md",))
print(f"wrote reports/Part1_audit.md, Report.md, CEO_report.md  -  {left} <!-- WRITE --> slots in Report.md still need YOUR interpretation")
