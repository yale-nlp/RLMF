import json
import csv
from pathlib import Path
import sys

def main(directory):
    dir_path = Path(directory)
    json_files = sorted(dir_path.glob("test_scores_*.json"))
    
    rows = []
    for json_file in json_files:
        # Extract * from filename
        filename_part = json_file.stem.replace("test_scores_", "")
        
        with open(json_file) as f:
            data = json.load(f)
        
        row = {
            "filename": filename_part,
            "cMFG*": data["cmfg_star_numeric"],
            "Var cMFG*": data["var_cmfg_star_numeric"],
            "cMFG": data["cmfg_numeric"],
            "MFG": data["mfg_numeric"],
            "std": data["stats_numeric"]["cmfg_std"],
            "sem": data["stats_numeric"]["cmfg_sem"],
            "Acc": data["avg_acc"],
            "B.S.": data["avg_bs"]
        }
        rows.append(row)

    desired_order = ["popqa", "selfaware", "math", "sciq", "simpleqa", "halueval", "umwp", "mmlu", "arc_challenge", "superglue"]
    order_map = {name: i for i, name in enumerate(desired_order)}
    rows.sort(key=lambda x: order_map.get(x["filename"], len(desired_order)))
    
    # Write CSV
    output_file = dir_path / "test_scores_all.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "cMFG*", "Var cMFG*", "cMFG", "MFG", "std", "sem", "Acc", "B.S."])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    main(sys.argv[1])