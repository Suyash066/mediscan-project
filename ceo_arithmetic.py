"""Part 2, task 7c: the arithmetic behind the CEO report. Pure maths from the constants in lib.py, plus the measured test results if present.
Usage: python ceo_arithmetic.py"""
import json
import numpy as np, pandas as pd
from lib import *

PREV, CFN, CFP = CAMP_PREVALENCE, COST_FN, COST_FP
def row(name, sens, spec, note=""):
    acc = PREV * sens + (1 - PREV) * spec; fn = PREV * (1 - sens) * 1000; fp = (1 - PREV) * (1 - spec) * 1000
    ppv = PREV * sens / (PREV * sens + (1 - PREV) * (1 - spec)) if (PREV * sens + (1 - PREV) * (1 - spec)) > 0 else float("nan")
    return dict(classifier=name, sensitivity=sens, specificity=spec, accuracy=acc, missed_per_1000=fn, false_alarms_per_1000=fp, PPV=ppv, expected_cost_per_patient=float(expected_cost(sens, 1 - spec)), note=note)
rows = [row("flags nobody", 0.0, 1.0), row("flags everybody", 1.0, 0.0),
        row("Rohit in the pilot (spec from pilot; sens ASSUMED 0.98)", 0.98, 0.30, "7/10 healthy flagged => spec 0.30"),
        row("a '99% accurate' model", 0.90, 1.0, "one way to reach 99%: miss 1 in 10 sick children"),
        row("a '95% accurate' model", 0.99, 0.95 + 0.0, "high sensitivity, some false alarms")]
tab = pd.DataFrame(rows); tab.to_csv(TABLES / "p2_ceo_arithmetic.csv", index=False)
# 98.4% on Rohit's test (234 NORMAL / 390 PNEUMONIA): interval and baseline
n, k = 624, int(round(0.984 * 624)); z = 1.959964; ph = k / n
wil = ((ph + z * z / (2 * n) - z * np.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)) / (1 + z * z / n), (ph + z * z / (2 * n) + z * np.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)) / (1 + z * z / n))
md = ["## Part 2.7 The arithmetic behind the CEO report\n",
      f"**Rohit's 98.4%.** On his test folder (234 NORMAL + 390 PNEUMONIA = {n} images, 62.5% pneumonia) a model that says 'pneumonia' for everybody already scores 62.5%; 98.4% is {k}/{n} with a 95% Wilson interval [{wil[0]:.3f}, {wil[1]:.3f}] "
      "**if the test images were independent of training**, which the Part-1 audit shows they are not (near-duplicates and shared patients across train/test), and drawn from one hospital. It measures memory of one site, not clinical skill. "
      "The pilot then showed specificity of about 30% (7 of 10 healthy children flagged) - invisible in an accuracy number dominated by the 62.5% pneumonia majority.\n",
      f"**Camp prevalence arithmetic** (per 1000 children, {PREV:.0%} sick = {1000*PREV:.0f} sick / {1000*(1-PREV):.0f} healthy; cost: miss = {CFN:.0f}, false alarm = {CFP:.0f}):\n",
      md_table(tab.drop(columns="note"), "{:.3f}"), "",
      f"A model that does nothing scores {1-PREV:.0%} accuracy. So '99% accuracy' is only 9 points above doing nothing, and it can be reached by missing 1 sick child in 10 (row 4: expected cost {tab.expected_cost_per_patient[3]:.3f} per patient) "
      f"while a '95%-accurate' model that catches 99% of sick children is **cheaper** ({tab.expected_cost_per_patient[4]:.3f}). Accuracy rewards the wrong thing here. "
      "**Ask for instead:** sensitivity (with a lower confidence bound) at a stated specificity, measured at the camp prevalence on an external site, plus calibration and the share of cases referred - i.e. minimise expected clinical cost, not maximise accuracy.\n"]
t = RESULTS / "threshold.json"
if t.exists():
    cost_tab = TABLES / "p2_cost_table.csv"
    if cost_tab.exists():
        c = pd.read_csv(cost_tab); r = c[c.rule.str.startswith("cost-optimal")].iloc[0]
        ours = row("OUR MODEL (test split, chosen threshold)", float(r.recall), float(r.specificity))
        md += ["**Our model at the chosen operating point, expressed at camp prevalence (per 1000 children):**\n", md_table(pd.DataFrame([ours]).drop(columns="note"), "{:.3f}"), ""]
write_fragment("p2_ceo", "\n".join(md))
