import os, json, sys
from pathlib import Path
import pandas as pd

def get_stats_df(data):

    confidence_bins = [i/10 for i in range(1, 11)]
    confidences, decisivenesses, faithfulnesses, bin_ids = [], [], [], []
    
    for c_list, d, f in zip(data['gold_confs_per_response'], data['avg_pred_confs'], data['f_scores']):

        if d==-1: continue
        c_list = [x for x in c_list if x!=-1]
        average_confidence = sum(c_list)/len(c_list)
        bin_idx = 0
        for i, bin_val in enumerate(confidence_bins):
            if average_confidence <= bin_val:
                bin_idx = i
                break
        bin_ids.append(bin_idx)
        confidences.append(average_confidence)
        decisivenesses.append(d)
        faithfulnesses.append(f)

    # Get the number of samples in each bin
    bin_counts = [bin_ids.count(i) for i in range(len(confidence_bins))]
    bin_idx_to_name = {i: f"({confidence_bins[i]-(1. / len(confidence_bins)):.1f}, {confidence_bins[i]:.1f}]" for i in range(1, len(confidence_bins))}
    bin_idx_to_name[0] = f"[0.0, {confidence_bins[0]:.1f}]"

    # Get MFG per bin
    MFG_per_bin = [0.0 for _ in range(len(confidence_bins))]
    for i in range(len(confidence_bins)):
        bin_faithfulnesses = [faithfulnesses[j] for j in range(len(faithfulnesses)) if bin_ids[j] == i]
        MFG_per_bin[i] = sum(bin_faithfulnesses)/len(bin_faithfulnesses) if len(bin_faithfulnesses)>0 else 0

    cMFG = sum(MFG_per_bin)/len(MFG_per_bin) 

    # Create per-bin stats
    df_data = []
    for i in range(len(confidence_bins)):
        bin_confidences = [confidences[j] for j in range(len(confidences)) if bin_ids[j] == i]
        mean_confidence = sum(bin_confidences)/len(bin_confidences) if len(bin_confidences)>0 else -1
        
        bin_decisivenesses = [decisivenesses[j] for j in range(len(decisivenesses)) if bin_ids[j] == i]
        mean_decisiveness = sum(bin_decisivenesses)/len(bin_decisivenesses) if len(bin_decisivenesses)>0 else -1
        
        df_data.append({
            'Bin': bin_idx_to_name[i],
            'Bin Counts': int(bin_counts[i]),
            'Mean Bin F': MFG_per_bin[i],
            'Mean Bin Conf': mean_confidence,
            'Mean Bin Dec': mean_decisiveness
        })

    # Add overall row
    df_data.append({
        'Bin': 'Overall',
        'Bin Counts': sum(bin_counts),
        'Mean Bin F': sum(faithfulnesses)/len(faithfulnesses),
        'Mean Bin Conf': sum(confidences)/len(confidences),
        'Mean Bin Dec': sum(decisivenesses)/len(decisivenesses)
    })
    # Add cMFG row
    df_data.append({
        'Bin': 'Final cMFG',
        'Bin Counts': None,
        'Mean Bin F': None,
        'Mean Bin Conf': None,
        'Mean Bin Dec': cMFG
    })
    # Add Orig stats rows
    df_data.append({
        'Bin': 'Orig Stats',
        'Bin Counts': f"cMFG_num",
        'Mean Bin F': f"MFG_num",
        'Mean Bin Conf': f"Acc",
        'Mean Bin Dec':  f"B.S.",
    })
    df_data.append({
        'Bin': None,
        'Bin Counts': f"{data['cmfg_numeric']:2f}",
        'Mean Bin F': f"{data['mfg_numeric']:2f}",
        'Mean Bin Conf': f"{data['avg_acc']:2f}",
        'Mean Bin Dec':  f"{data['avg_bs']:2f}",
    })

    # Create DataFrame and save
    df = pd.DataFrame(df_data)
    print(df)

    return df

def main(results_dir):

    dir_path = Path(results_dir)
    json_files = sorted(dir_path.glob("test_scores_*.json"))
    if len(json_files)==0:
        json_files = sorted(dir_path.glob("test_scores*.json"))
    
    rows = []
    for json_file in json_files:
        # Extract * from filename
        dataset_name = json_file.stem.replace("test_scores_", "")
        if dataset_name=="test_scores": dataset_name = results_dir.split("_baseline_")[-1].split("_")[0]
        
        with open(json_file) as f:
            data = json.load(f)
        
        stats_df = get_stats_df(data)
    
        # Write CSV
        csv_path = dir_path / f"conf_bin_analysis_{dataset_name}.csv"
        stats_df.to_csv(csv_path, index=False)

        print(f"\nSaved {dataset_name} results to {csv_path}")
    

if __name__ == "__main__":
    main(sys.argv[1])

