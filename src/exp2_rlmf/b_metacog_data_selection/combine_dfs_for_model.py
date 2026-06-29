#!/usr/bin/env python3
"""
Script to combine score CSV files from a specific model directory.
Combines files matching pattern: scores_aggr_over_sampled_answers_popqa_train_*.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--pattern", type=str, default="scores_aggr_over_sampled_answers")
    parser.add_argument("--output_identifier", type=str, default="")
    parser.add_argument( "--base_path", type=str, default="./exp2_rlmf/b_metacog_data_selection/score_dfs", help="Scores df directory path")
    args = parser.parse_args()
    
    model_dir = Path(args.base_path) / args.model_name.replace("-", "_").replace("/", "_")
    os.makedirs(str(model_dir), exist_ok=True)
    
    # Check if directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find all matching CSV files
    # pattern = f"scores_aggr_over_sampled_answers_{args.dataset_name}_train_*.csv"
    pattern = f"{args.pattern}_{args.dataset_name}_train_*.csv"
    csv_files = sorted(model_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching pattern '{pattern}' found in {model_dir}"
        )
    print(f"Found {len(csv_files)} files to combine:")
    
    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"Loaded {csv_file.name}: {len(df)} rows")
    
    # Vertically stack (concatenate) all dataframes
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"\nCombined DataFrame: {len(combined_df)} total rows")
    
    # Save to CSV
    output_file = os.path.join(model_dir, f"_train{args.output_identifier}_{args.dataset_name}.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined CSV to: {output_file}")
