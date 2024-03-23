from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
)

from trainers.train_vqvae import train_vqvae
from trainers.train_ssl_vqvae import train_ssl_vqvae
from trainers.train_maskgit import train_maskgit


import torch

STAGE1_PROJECT_NAME = "SpringBreak Hallelujah stage 1"
STAGE2_PROJECT_NAME = "SpringBreak Hallelujah stage 2"

UCR_SUBSET = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    # "TwoPatterns",
    "FordA",
    "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]

STAGE1_EPOCHS = 1000
STAGE2_EPOCHS = 1000

NUM_RUNS_PER = 1

SSL_METHODS = ["", "vibcreg", "barlowtwins"]

RUN_STAGE1 = True
RUN_STAGE2 = True

SEED = 0


def run_experiments():
    # Set manual seed
    torch.manual_seed(SEED)

    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)
    # Set max epochs for each stage
    config["trainer_params"]["max_epochs"]["stage1"] = STAGE1_EPOCHS
    config["trainer_params"]["max_epochs"]["stage2"] = STAGE2_EPOCHS
    # Only reconstruct original view:
    config["VQVAE"]["recon_augmented_view_scale"] = 0.0
    config["VQVAE"]["recon_original_view_scale"] = 1.0

    batch_size_stage1 = config["dataset"]["batch_sizes"]["stage1"]
    batch_size_stage2 = config["dataset"]["batch_sizes"]["stage2"]

    # List of experiments to run
    experiments = []

    if RUN_STAGE1:
        experiments += [
            # Stage 1
            {
                "stage": 1,
                "ssl_method": method,
                "augmented_data": (method != ""),
                "orthogonal_reg_weight": ortho_reg,
                "project_name": STAGE1_PROJECT_NAME,
                "train_fn": train_vqvae if method == "" else train_ssl_vqvae,
            }
            for ortho_reg in [0, 10]
            for method in SSL_METHODS
        ]

    if RUN_STAGE2:
        experiments += [
            # Stage 2
            {
                "stage": 2,
                "ssl_method": method,
                "augmented_data": False,
                "orthogonal_reg_weight": 0,  # ortho_reg,
                "project_name": STAGE2_PROJECT_NAME,
                "train_fn": train_maskgit,
            }
            # for ortho_reg in [0, 10]
            # for method in SSL_METHODS
        ]

    print("Experiments to run:")
    for i, exp in enumerate(experiments):
        print(f"{i+1}. {exp}\n")

    for dataset in UCR_SUBSET:
        c = config.copy()
        c["dataset"]["dataset_name"] = dataset

        # Build data pipelines
        dataset_importer = UCRDatasetImporter(**c["dataset"])
        train_data_loader_stage1 = build_data_pipeline(
            batch_size_stage1, dataset_importer, c, augment=False, kind="train"
        )
        train_data_loader_stage1_aug = build_data_pipeline(
            batch_size_stage1, dataset_importer, c, augment=True, kind="train"
        )
        train_data_loader_stage2 = build_data_pipeline(
            batch_size_stage2, dataset_importer, c, augment=False, kind="train"
        )
        test_data_loader = build_data_pipeline(
            batch_size_stage1, dataset_importer, c, augment=False, kind="test"
        )  # Same test dataloader for both stages
        # Running experiments:
        for experiment in experiments:
            # Only configure stage 1 method:
            c["SSL"][f"stage1_method"] = experiment["ssl_method"]
            c["VQVAE"]["orthogonal_reg_weight"] = experiment["orthogonal_reg_weight"]

            for run in range(NUM_RUNS_PER):
                # Wandb run name:
                method = experiment["ssl_method"]
                decorr = "decorr-" if experiment["orthogonal_reg_weight"] > 0 else ""
                ssl_method = f"{method}-" if method != "" else ""
                stage = "stage1" if experiment["stage"] == 1 else "stage2"

                run_name = f"{decorr}{ssl_method}{stage}"

                # Set correct data loader
                if experiment["stage"] == 1:
                    train_data_loader = (
                        train_data_loader_stage1_aug
                        if experiment["augmented_data"]
                        else train_data_loader_stage1
                    )
                else:
                    train_data_loader = train_data_loader_stage2

                experiment["train_fn"](
                    config=c,
                    train_data_loader=train_data_loader,
                    test_data_loader=test_data_loader,
                    do_validate=True,
                    gpu_device_idx=0,
                    wandb_run_name=f"{run_name}-{dataset}-run{run+1}",
                    wandb_project_name=experiment["project_name"],
                    torch_seed=SEED,
                )


if __name__ == "__main__":
    run_experiments()
