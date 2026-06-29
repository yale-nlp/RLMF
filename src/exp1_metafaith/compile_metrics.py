"""
Usage: 
python ./exp1_metafaith/compile_metrics.py --dir ./exp1_metafaith/_results/model_name_with_underscores

Example:
python ./exp1_metafaith/compile_metrics.py --dir ./exp1_metafaith/_results/gemini_3_flash_preview

"""
import json
import os
import argparse
import pandas as pd

DATASETS = ["popqa", "selfaware", "simpleqa", "halueval", "mmlu", "sciq", "math", "umwp", "arc_challenge", "superglue"]
METRICS = ["cmfg_star", "cmfg", "mfg", "avg_acc", "avg_bs"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    data = {m: {} for m in METRICS}

    for dataset in DATASETS:
        # find matching file
        matches = [f for f in os.listdir(args.dir) if f"scores_sys0_{dataset}_" in f and f.endswith(".json")]
        if not matches:
            print(f"No file found for {dataset}, skipping.")
            continue
        path = os.path.join(args.dir, matches[0])
        with open(path) as f:
            scores = json.load(f)
        for metric in METRICS:
            data[metric][dataset] = scores.get(metric, None)

    df = pd.DataFrame(data).T
    df.index.name = "Metric"
    df = df[DATASETS]
    df["Average"] = df.mean(axis=1)

    out_path = os.path.join(args.dir, "_scores_compiled.csv")
    df.to_csv(out_path, index=True)
    print(df.to_string(index=True))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()

