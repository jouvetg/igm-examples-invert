import os
import glob
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# CONFIGURATION
BASE_DIR = ""  # Replace with your actual output base directory
METRIC_FILES = ["rms_std.dat", "costs.dat"]  # List of metric files per run
OPTIMIZE_METRICS = []  # Leave empty to choose interactively

def parse_metrics(run_dir):
    dfs = []
    metric_names = []

    for file_name in METRIC_FILES:
        file_path = os.path.join(run_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"Warning: missing {file_path}")
            return None, []

        # Read full file (assumes header is first row)
        df = pd.read_csv(file_path, delim_whitespace=True, header=0)
        if df.empty:
            print(f"Warning: empty file {file_path}")
            return None, []

        dfs.append(df)
        metric_names += list(df.columns)

    # Combine all metrics horizontally, like `paste`
    merged_df = pd.concat(dfs, axis=1)

    # Return last row only
    final_metrics = merged_df.iloc[-1].to_dict()
    return final_metrics, metric_names

def parse_overrides(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            overrides = yaml.safe_load(f)
        if isinstance(overrides, list):
            params = {}
            for item in overrides:
                if "=" in item:
                    key, val = item.split("=", 1)
                    params[key.strip()] = val.strip()
            return params
        elif isinstance(overrides, dict):
            return overrides
        return {}
    except Exception as e:
        print(f"Error reading overrides.yaml: {e}")
        return {}

def collect_all_runs():
    data = []
    all_metrics = set()

    run_dirs = sorted(glob.glob(os.path.join(BASE_DIR, '*')))
    for run_dir in run_dirs:
        yaml_path = os.path.join(run_dir, '.hydra', 'overrides.yaml')
        if not os.path.isfile(yaml_path):
            continue

        params = parse_overrides(yaml_path)
        metrics, names = parse_metrics(run_dir)
        if metrics is None:
            continue

        all_metrics.update(names)

        entry = {'run_dir': run_dir}
        entry.update(params)
        entry.update(metrics)
        data.append(entry)

    df = pd.DataFrame(data)
    return df, sorted(all_metrics)
 
if __name__ == "__main__":
    df, metric_names = collect_all_runs()

    if df.empty:
        print("No valid runs found.")
    else:
        # Show all output fully
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 0)
        pd.set_option('display.max_colwidth', None)

        print(f"Found {len(df)} runs with metrics: {metric_names}")
        print(df)

        df.to_csv("summary_metrics.csv", index=False)

 
