import json
import os
import argparse
import pandas as pd

DATASETS = ["popqa", "selfaware", "simpleqa", "halueval", "mmlu", "sciq", "math", "umwp", "arc_challenge", "superglue"]
METRICS = [
    "cmfg_linguistic_with_assertions",
    "mfg_linguistic_with_assertions",
    "cmfg_linguistic_without_assertions",
    "mfg_linguistic_without_assertions",
    "cmfg_star_linguistic_with_assertions",
    "var_cmfg_star_with_assertions",
    "cmfg_star_linguistic_without_assertions",
    "var_cmfg_star_without_assertions",
    "avg_acc",
    "avg_bs"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--bin_size", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model_str = args.model
    mode_str = args.mode
    bin_str = args.bin_size

    data = {m: {} for m in METRICS}

    for dataset in DATASETS:
        pattern = f"linguistic_scores_{dataset}_{mode_str}_{bin_str}_{model_str}.json"
        path = os.path.join(args.dir, pattern)
        if not os.path.exists(path):
            print(f"File not found for {dataset}: {pattern}, skipping.")
            print(path)
            continue
        with open(path) as f:
            scores = json.load(f)
        for metric in METRICS:
            data[metric][dataset] = scores.get(metric, None)

    df = pd.DataFrame(data).T
    df.index.name = "Metric"
    available = [d for d in DATASETS if d in df.columns]
    df = df[available]
    df["Average"] = df.mean(axis=1)

    out_name = f"_scores_compiled_{mode_str}_{bin_str}_{args.model}.csv"
    out_path = os.path.join(args.dir, out_name)
    df.to_csv(out_path, index=True)
    print(df.to_string(index=True))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
