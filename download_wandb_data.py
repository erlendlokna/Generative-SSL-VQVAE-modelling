import pandas as pd
import os
from tqdm import tqdm
import wandb

filters = {
    "regular": {
        "config.SSL.stage1_method": "",
        "config.VQVAE.orthogonal_reg_weight": 0,
    },
    "regular-decorr": {
        "config.SSL.stage1_method": "",
        "config.VQVAE.orthogonal_reg_weight": 10,
    },
    "vibcreg": {
        "config.SSL.stage1_method": "vibcreg",
        "config.VQVAE.orthogonal_reg_weight": 0,
    },
    "vibcreg-decorr": {
        "config.SSL.stage1_method": "vibcreg",
        "config.VQVAE.orthogonal_reg_weight": 10,
    },
    "barlowtwins": {
        "config.SSL.stage1_method": "barlowtwins",
        "config.VQVAE.orthogonal_reg_weight": 0,
    },
    "barlowtwins-decorr": {
        "config.SSL.stage1_method": "barlowtwins",
        "config.VQVAE.orthogonal_reg_weight": 10,
    },
}


def wandb_stage1_scan_to_csv(wandb_stage1_project, dataset, api=wandb.Api()):
    # Metrics we're interested in logging
    metrics_keys = [
        "loss",
        "ssl_loss",
        "svm_accuracy",
        "knn_accuracy",
        "perplexity",
        "val_perplexity",
        "mean_abs_corr_off_diagonal",
        "mean_abs_cos_sim_off_diagonal",
        "val_loss",
        "training_time",
    ]

    # Ensure the dataset name is included in each filter
    for f in filters.values():
        f["config.dataset.dataset_name"] = dataset

    root_dir = f"results/{dataset}/stage1"
    os.makedirs(root_dir, exist_ok=True)

    for key, filter in tqdm(filters.items()):
        runs = api.runs(wandb_stage1_project, filters=filter)

        # Initialize a list to hold all the metrics for all runs
        all_metrics = []

        for run in runs:
            history = run.scan_history()
            # For each record in the history, only capture the metrics of interest
            for record in history:
                row = {metric: record.get(metric, None) for metric in metrics_keys}
                row["run_id"] = run.id  # Keep track of the run ID
                all_metrics.append(row)

        # Convert the accumulated metrics to a DataFrame
        df_metrics = pd.DataFrame(all_metrics)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)

    print(f"Metrics written to {root_dir}")


def wandb_stage1_summary_to_csv(wandb_stage1_project, dataset, api=wandb.Api()):

    metrics_keys = [
        "loss",
        "ssl_loss",
        "svm_accuracy",
        "knn_accuracy",
        "perplexity",
        "val_perplexity",
        "mean_abs_corr_off_diagonal",
        "mean_abs_cos_sim_off_diagonal",
        "val_loss",
        "training_time",
    ]

    # Ensure the dataset name is included in each filter
    for f in filters.values():
        f["config.dataset.dataset_name"] = dataset

    root_dir = f"results/{dataset}/stage1"
    os.makedirs(root_dir, exist_ok=True)

    for key, filter in tqdm(filters.items()):
        runs = api.runs(wandb_stage1_project, filters=filter)

        # Initialize a list to hold all the summaries for all runs
        all_summaries = []

        for run in runs:
            # For each run, capture the summary
            summary = run.summary
            row = {metric: summary.get(metric, None) for metric in metrics_keys}
            row["run_id"] = run.id  # Keep track of the run ID
            all_summaries.append(row)

        # Convert the accumulated summaries to a DataFrame
        df_summaries = pd.DataFrame(all_summaries)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_summaries.csv")
        df_summaries.to_csv(csv_path, index=False)

    print(f"Summaries written to {root_dir}")


def wandb_stage2_scans_to_csv(wandb_stage2_project, dataset, api=wandb.Api()):
    # Metrics we're interested in logging
    metrics_keys = [
        "kurtosis",
        "coverage",
        "entropy",
        "FID",
        "IS_mean",
        "prior_loss",
        "IS_std",
        "skewness",
        "variety (Gini)",
    ]

    # Ensure the dataset name is included in each filter
    for f in filters.values():
        f["config.dataset.dataset_name"] = dataset

    root_dir = f"results/{dataset}/stage2"
    os.makedirs(root_dir, exist_ok=True)

    for key, filter in tqdm(filters.items()):
        runs = api.runs(wandb_stage2_project, filters=filter)

        # Initialize a list to hold all the metrics for all runs
        all_metrics = []

        for run in runs:
            history = run.scan_history()
            # For each record in the history, only capture the metrics of interest
            for record in history:
                row = {metric: record.get(metric, None) for metric in metrics_keys}
                row["run_id"] = run.id  # Keep track of the run ID
                all_metrics.append(row)

        # Convert the accumulated metrics to a DataFrame
        df_metrics = pd.DataFrame(all_metrics)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)

    print(f"Metrics written to {root_dir}")


def wandb_stage2_summary_to_csv(wandb_stage2_project, dataset, api=wandb.Api()):
    metrics_keys = [
        "kurtosis",
        "coverage",
        "entropy",
        "FID",
        "IS_mean",
        "prior_loss",
        "IS_std",
        "skewness",
        "variety (Gini)",
    ]

    # Ensure the dataset name is included in each filter
    for f in filters.values():
        f["config.dataset.dataset_name"] = dataset

    root_dir = f"results/{dataset}/stage2"
    os.makedirs(root_dir, exist_ok=True)

    for key, filter in tqdm(filters.items()):
        runs = api.runs(wandb_stage2_project, filters=filter)

        # Initialize a list to hold all the summaries for all runs
        all_summaries = []

        for run in runs:
            # For each run, capture the summary
            summary = run.summary
            row = {metric: summary.get(metric, None) for metric in metrics_keys}
            row["run_id"] = run.id  # Keep track of the run ID
            all_summaries.append(row)

        # Convert the accumulated summaries to a DataFrame
        df_summaries = pd.DataFrame(all_summaries)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_summaries.csv")
        df_summaries.to_csv(csv_path, index=False)

    print(f"Summaries written to {root_dir}")


datasets = [
    "ElectricDevices",
    "StarLightCurves",
    "Wafer",
    "ECG5000",
    "TwoPatterns",
    "FordA",
    "UWaveGestureLibraryAll",
    "FordB",
    "ShapesAll",
]

if __name__ == "__main__":
    wandb_stage1_proj = "S1-Master-Run"
    wandb_stage2_proj = "S2-Master-Run"

    for dataset in datasets:
        # Stage 1
        wandb_stage1_summary_to_csv(wandb_stage1_proj, dataset)

        # Stage 2
        wandb_stage2_summary_to_csv(wandb_stage2_proj, dataset)
        # wandb_stage2_runs_to_csv(wandb_stage2_proj, dataset)
