import os
import re
import json
import argparse
import pandas as pd
from termcolor import colored

CHECKPOINT_REGEX = re.compile(r"^checkpoint_(\d+)$")

FIELDS = [
    "cmfg_numeric",
    "mfg_numeric",
    "cmfg_star_numeric",
    "var_cmfg_star_numeric",
    "avg_acc",
    "avg_bs",
    "avg_lin_reward",
    "avg_quad_reward",
    "avg_bin_reward",
    "avg_simp_log_reward",
    "avg_stret_log_reward",
]

def main(model_dir: str, dataset_name: str):
    rows = []

    test_results_dir = os.path.join(model_dir, "test_results")
    for name in os.listdir(test_results_dir):
        match = CHECKPOINT_REGEX.match(name)
        if not match:
            continue

        checkpoint_num = int(match.group(1))
        ckpt_dir = os.path.join(test_results_dir, name)
        if dataset_name is not None:
            scores_path = os.path.join(ckpt_dir, f"test_scores_{dataset_name}.json")
            if not os.path.isfile(scores_path):
                print(colored("[WARN]","red") + f" Missing test_scores_{dataset_name}.json in {ckpt_dir}")
                continue
            else: 
                with open(scores_path, "r") as f:
                    scores = json.load(f)
        else:
            scores_path = os.path.join(ckpt_dir, f"test_scores.json")
            scores_path_other = os.path.join(ckpt_dir, f"test_scores_popqa.json")
            if not os.path.isfile(scores_path) and not os.path.isfile(scores_path_other):
                print(colored("[WARN]","red") + f" Missing test_scores.json and test_scores_popqa.json in {ckpt_dir}")
                continue
            if os.path.isfile(scores_path):
                with open(scores_path, "r") as f:
                    scores = json.load(f)
            elif os.path.isfile(scores_path_other):
                with open(scores_path_other, "r") as f:
                    scores = json.load(f)

        row = {"checkpoint": checkpoint_num}

        # top-level fields
        for field in FIELDS:
            row[field] = scores.get(field)

        # nested stats_numeric fields
        stats_numeric = scores.get("stats_numeric", {})
        row["cmfg_std"] = stats_numeric.get("cmfg_std")
        row["cmfg_sem"] = stats_numeric.get("cmfg_sem")

        rows.append(row)

    if not rows:
        raise RuntimeError("No valid checkpoint_* directories found.")

    df = pd.DataFrame(rows).sort_values("checkpoint")

    column_order = ["checkpoint", "cmfg_star_numeric", "var_cmfg_star_numeric", "cmfg_numeric", "mfg_numeric", "cmfg_std", "cmfg_sem", "avg_acc", "avg_bs", "avg_lin_reward", "avg_quad_reward", "avg_bin_reward", "avg_simp_log_reward", "avg_stret_log_reward"]

    df = df[column_order]

    if dataset_name is not None:
        output_path = os.path.join(model_dir, f"_stats_summary_{dataset_name}.csv")
    else:
        output_path = os.path.join(model_dir, "_stats_summary.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved summary CSV to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_results_dir",
        type=str,
        required=True,
        help="Path to test_results directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    main(args.test_results_dir, args.dataset_name)
