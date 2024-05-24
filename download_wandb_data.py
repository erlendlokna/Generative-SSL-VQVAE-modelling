import pandas as pd
import os
from tqdm import tqdm
import wandb

filters = {
    "regular": {
        "config.SSL.stage1_method": "",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["window_warp", "amplitude_resize"],
    },
    "vibcreg_warp": {
        "config.SSL.stage1_method": "vibcreg",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["window_warp", "amplitude_resize"],
    },
    "vibcreg_slice": {
        "config.SSL.stage1_method": "vibcreg",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["slice_and_shuffle"],
    },
    "vibcreg_gauss": {
        "config.SSL.stage1_method": "vibcreg",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["gaussian_noise"],
    },
    "barlowtwins_warp": {
        "config.SSL.stage1_method": "barlowtwins",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["window_warp", "amplitude_resize"],
    },
    "barlowtwins_slice": {
        "config.SSL.stage1_method": "barlowtwins",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["slice_and_shuffle"],
    },
    "barlowtwins_gauss": {
        "config.SSL.stage1_method": "barlowtwins",
        "config.VQVAE.orthogonal_reg_weight": 0,
        "config.augmentations.time_augs": ["gaussian_noise"],
    },
}

metrics_keys1 = [
    "svm_accuracy",
    "knn_accuracy",
    "perplexity",
    "val_perplexity",
    "mean_abs_corr_off_diagonal",
    "mean_abs_cos_sim_off_diagonal",
    "val_loss",
    "training_time",
]


metrics_keys2 = [
    "FID",
    "IS_mean",
    "IS_std",
]


datasets = [
    "ElectricDevices",
    # "FordB",
    "FordA",
    "Wafer",
    "TwoPatterns",
    "StarLightCurves",
    "UWaveGestureLibraryAll",
    "ECG5000",
    # "ShapesAll",
    "Mallat",
    "Symbols",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
]


def wandb_stage1_summary_to_csv(wandb_stage1_project, dataset, api=wandb.Api()):

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
            row = {metric: summary.get(metric, None) for metric in metrics_keys1}
            row["run_id"] = run.id  # Keep track of the run ID
            all_summaries.append(row)

        # Convert the accumulated summaries to a DataFrame
        df_summaries = pd.DataFrame(all_summaries)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_summaries.csv")
        df_summaries.to_csv(csv_path, index=False)

    print(f"Summaries written to {root_dir}")


def wandb_stage2_summary_to_csv(wandb_stage2_project, dataset, api=wandb.Api()):

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
            row = {metric: summary.get(metric, None) for metric in metrics_keys2}
            row["run_id"] = run.id  # Keep track of the run ID
            all_summaries.append(row)

        # Convert the accumulated summaries to a DataFrame
        df_summaries = pd.DataFrame(all_summaries)

        # Write the DataFrame to a CSV file
        csv_path = os.path.join(root_dir, f"{key}_summaries.csv")
        df_summaries.to_csv(csv_path, index=False)

    print(f"Summaries written to {root_dir}")


if __name__ == "__main__":
    wandb_stage1_proj = "S1-Augs"
    wandb_stage2_proj = "S2-Augs"

    for dataset in datasets:
        # Stage 1
        wandb_stage1_summary_to_csv(wandb_stage1_proj, dataset)

        # Stage 2
        wandb_stage2_summary_to_csv(wandb_stage2_proj, dataset)
        # wandb_stage2_runs_to_csv(wandb_stage2_proj, dataset)
