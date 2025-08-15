#!/usr/bin/env python3
# ------------------------------------------------------------
# Evaluate VRU multiple-choice predictions
# ------------------------------------------------------------
import json, pathlib, re
from collections import defaultdict

# ---------- file names --------------------------------------------------------
GT_FILE   = "cliped_test_pilot.json"      # ground truth (full, with GT fields)
PRED_FILE = "cliped_vru_predictions.json"   # your model’s answers
# ------------------------------------------------------------------------------

QUESTION_ORDER = [
    "weather and light",
    "location",
    "road type",
    "accident type",
    "accident reason",
    "prevention method"
]

def load_ground_truth(fp: str):
    """return dict: video -> list[str] (6 GT letters)"""
    with open(fp, "r", encoding="utf8") as f:
        meta = json.load(f)
    out = {}
    for vid, qa in meta.items():
        out[vid] = [qa[q]["GT"].strip().upper() for q in QUESTION_ORDER]
    return out

def load_predictions(fp: str):
    """return dict: video -> list[str] (6 predicted letters)"""
    with open(fp, "r", encoding="utf8") as f:
        raw = json.load(f)

    cleaned = {}
    for vid, answers in raw.items():
        # answers might be ["A\nB\nC\nD\nE\nF"]  OR  ["A","B",...]
        if len(answers) == 1:
            # split on anything that is not A–D
            letters = re.findall(r"[ABCD]", answers[0].upper())
        else:
            letters = [re.search(r"[ABCD]", x.upper()).group(0)  # first valid letter
                       if re.search(r"[ABCD]", x.upper()) else "?"
                       for x in answers]
        # pad / truncate to 6
        letters = (letters + ["?"]*6)[:6]
        cleaned[vid] = letters
    return cleaned

def evaluate(gt, pred):
    per_q_correct = defaultdict(int)
    per_q_total   = defaultdict(int)
    total_correct = total = 0
    details = []

    for vid in gt:
        if vid not in pred:
            print(f"⚠️  missing prediction for {vid}")
            continue
        g, p = gt[vid], pred[vid]
        row = [vid]
        for i,(gg,pp) in enumerate(zip(g,p)):
            ok = gg == pp
            row.append(f"{pp}{'✓' if ok else '✗'}")
            per_q_total[i]+=1
            if ok: 
                per_q_correct[i]+=1
                total_correct +=1
            total +=1
        details.append(row)

    # summary
    global_acc = total_correct/total if total else 0
    per_q_acc  = {QUESTION_ORDER[i]: per_q_correct[i]/per_q_total[i]
                  for i in range(len(QUESTION_ORDER))}

    return global_acc, per_q_acc, details

def pretty_print(global_acc, per_q_acc, details):
    print("\nGlobal accuracy: {:.2%}".format(global_acc))
    print("\nPer-question accuracy:")
    for k,v in per_q_acc.items():
        print(f"  {k:<20}: {v:.2%}")

    # make a tiny table
    header = ["video"] + [q.split()[0] for q in QUESTION_ORDER]  # short col names
    col_w  = [max(len(r[i]) for r in [header]+details) for i in range(len(header))]
    fmt    = "  ".join(f"{{:{w}}}" for w in col_w)

    print("\nDetails")
    print(fmt.format(*header))
    for r in details:
        print(fmt.format(*r))

def main():
    gt   = load_ground_truth(GT_FILE)
    pred = load_predictions(PRED_FILE)
    global_acc, per_q_acc, details = evaluate(gt, pred)
    pretty_print(global_acc, per_q_acc, details)

if __name__ == "__main__":
    main()