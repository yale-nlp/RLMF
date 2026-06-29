"""
Recomputes cMFG*, cMFG, MFG, STD, SEM, Acc, and BS for all datasets
and writes a summary CSV to --results_dir/test_scores_all_FINAL.csv.

Usage:
    python rescore.py --results_dir /path/to/results
"""

import os
import re
import json
import string
import argparse
import warnings

import numpy as np
import pandas as pd

from src.exp0_baseline.utilities.utils import get_cmfg_star, llm_eval
from src.exp2_rlmf.utils.utils import extract_sentences_with_confidence_new

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = [
    "popqa",
    "selfaware",
    "simpleqa",
    "sciq",
    "math",
    "umwp",
    "halueval",
    "arc_challenge",
    "superglue",
    "mmlu",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)

def run_extract_sentences_with_confidence(
    extracted_responses,
    get_meta_score_from_completions: bool = False,
    metascore_as_percentage: bool = False,
):
    """
    Parse each raw response into (sentences, confidences, metascore).
    Returns three parallel lists-of-lists.
    """
    sentences_per_completion = []
    pred_confs_per_completion = []
    metascore_per_completion = []

    for response in extracted_responses:
        result = extract_sentences_with_confidence_new(
            response,
            get_meta_score=get_meta_score_from_completions,
            metascore_as_percentage=metascore_as_percentage,
        )
        sentences, confidences = result[0], result[1]
        metascore = result[2] if get_meta_score_from_completions else None

        if len(sentences) == 0:
            sentences_per_completion.append(["no output"])
            pred_confs_per_completion.append([None])
            metascore_per_completion.append([None])
        else:
            sentences_per_completion.append(sentences)
            pred_confs_per_completion.append(confidences)
            metascore_per_completion.append(metascore)

    return sentences_per_completion, pred_confs_per_completion, metascore_per_completion

# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset: str, results_dir: str) -> dict | None:
    """
    Load score/pred files for one dataset, compute all metrics, return a row dict.
    Returns None if files are missing (with a warning).
    """
    scores_path = os.path.join(results_dir, f"test_scores_{dataset}.json")
    preds_path  = os.path.join(results_dir, f"test_preds_{dataset}.json")

    # --- guard: missing files ---
    missing = [p for p in (scores_path, preds_path) if not os.path.exists(p)]
    if missing:
        warnings.warn(
            f"[{dataset}] Skipping — file(s) not found: {missing}",
            stacklevel=2,
        )
        return None

    with open(scores_path) as f:
        scores_dict = json.load(f)
    with open(preds_path) as f:
        preds_dict = json.load(f)

    # ------------------------------------------------------------------
    # 1. Gold confidences
    # ------------------------------------------------------------------
    gold_confs_per_response = scores_dict["gold_confs_per_response"]

    avg_gold_confidences = [
        -1.0 if not (vals := [c for c in conf_list if c is not None]) else np.mean(vals)
        for conf_list in gold_confs_per_response
    ]
    all_avg_gold_confidences = np.array(avg_gold_confidences, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Extract sentences / strip confidence text from predicted answers
    # ------------------------------------------------------------------
    extracted_responses = preds_dict["answers"]
    sentences_per_response, pred_confs_per_response, metascore_per_response = (
        run_extract_sentences_with_confidence(extracted_responses)
    )

    # Reassemble cleaned predictions (confidence markers removed)
    responses_without_confidences = [
        " ".join(sent_list) for sent_list in sentences_per_response
    ]

    # ------------------------------------------------------------------
    # 3. Accuracy via LLM eval
    #    pred = stripped response (no confidence scores)
    # ------------------------------------------------------------------
    print(f"  [{dataset}] Running LLM-as-a-Judge acc scoring on {len(preds_dict['targets'])} examples...")
    accs = []

    for idx, (targets, pred) in enumerate(zip(preds_dict["targets"], responses_without_confidences)):
        accs.append(llm_eval(targets, pred))
        if idx%100==0: print(f"    Finished index {idx}!")

    all_accs = np.array(accs, dtype=np.float64)
    avg_acc  = float(all_accs.mean())

    # ------------------------------------------------------------------
    # 4. Brier scores
    # ------------------------------------------------------------------
    brier_scores = (all_accs - all_avg_gold_confidences) ** 2.0
    mask = all_avg_gold_confidences == -1.0
    brier_scores[mask] = -1.0
    valid_brier = brier_scores[brier_scores != -1.0]
    avg_bs = float(valid_brier.mean()) if len(valid_brier) > 0 else float("nan")

    # ------------------------------------------------------------------
    # 5. cMFG* via get_cmfg_star
    # ------------------------------------------------------------------
    faithfulness_scores = scores_dict["f_scores"]
    cmfg_star, var_cmfg_star, bin_info = get_cmfg_star(
        faithfulness_scores,
        avg_gold_confidences,   # plain Python list, as computed above
        num_bins=10,
    )

    # ------------------------------------------------------------------
    # 6. cMFG and MFG (precomputed in scores file)
    # ------------------------------------------------------------------
    cmfg = scores_dict["cmfg_numeric"]
    mfg  = scores_dict["mfg_numeric"]

    return {
        "Dataset":  dataset,
        "cMFG*":    cmfg_star,
        "cMFG":     cmfg,
        "MFG":      mfg,
        "cMFG* Var":   var_cmfg_star,
        "Acc":      avg_acc,
        "BS":       avg_bs,
    }, bin_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing test_scores_<dataset>.json and test_preds_<dataset>.json files.")
    args = parser.parse_args()

    out_path = os.path.join(args.results_dir, "test_scores_all_FINAL.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        if "Dataset" in df.columns:
            try:
                if "Average" not in df["Dataset"].values:
                    avg_row = df[["cMFG*", "cMFG", "MFG", "cMFG* Var", "Acc", "BS"]].mean()
                    avg_row["Dataset"] = "Average"
                    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)
            except:
                pass
            try:
                df = df.set_index("Dataset").T
            except:
                pass
        else:
            df = pd.read_csv(out_path, index_col=0)
            if "Average" not in df.columns:
                df["Average"] = df.mean(axis=1)
        df = df[["popqa", "selfaware", "simpleqa", "halueval", "mmlu", "sciq", "math", "umwp", "arc_challenge", "superglue", "Average"]]
        print(df.to_string(index=True))
        df.to_csv(out_path, index=True)
        print(f"\nSaved results to: {out_path}")
        return

    rows = []
    bin_infos = {}
    for dataset in DATASETS:
        print(f"\n{'='*60}\nProcessing: {dataset}\n{'='*60}")
        result = process_dataset(dataset, args.results_dir)
        if result is None:
            continue
        row, bin_info = result
        if row is not None:
            rows.append(row)
            bin_infos[dataset] = bin_info
            print(f"  [{dataset}] Done. Acc={row['Acc']:.4f}, BS={row['BS']:.4f}, cMFG*={row['cMFG*']:.4f}")

    if not rows:
        print("\nNo datasets processed successfully — nothing to save.")
        return

    df = pd.DataFrame(rows, columns=["Dataset", "cMFG*", "cMFG", "MFG", "cMFG* Var", "Acc", "BS"])

    avg_row = df[["cMFG*", "cMFG", "MFG", "cMFG* Var", "Acc", "BS"]].mean()
    avg_row["Dataset"] = "Average"
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    if "Average" not in df["Dataset"].values:
        avg_row = df[["cMFG*", "cMFG", "MFG", "cMFG* Var", "Acc", "BS"]].mean()
        avg_row["Dataset"] = "Average"
        df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    df = df.set_index("Dataset").T
    cols = ["popqa", "selfaware", "simpleqa", "halueval", "mmlu", "sciq", "math", "umwp", "arc_challenge", "superglue"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(out_path, index=True)
    print(f"\nSaved results to: {out_path}")
    print(df.to_string(index=True))

    bin_info_out_path = os.path.join(args.results_dir, "test_scores_bin_infos.json")
    with open(bin_info_out_path, "w") as f:
        json.dump(bin_infos, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()