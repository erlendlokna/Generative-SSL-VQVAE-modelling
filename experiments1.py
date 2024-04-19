from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
    generate_short_id,
    experiment_name,
)

from trainers.train_vqvae import train_vqvae
from trainers.train_ssl_vqvae import train_ssl_vqvae
from trainers.train_maskgit import train_maskgit
import torch


# Wandb logging information
STAGE1_PROJECT_NAME = "S1-Messiah-Run"
STAGE2_PROJECT_NAME = "S2-Messiah-Run"

# Stage 1 experiments to run
STAGE1_EXPS = ["", "vibcreg", "barlowtwins"]  # empty string means regular VQVAE
# Datasets to run experiments on
UCR_SUBSET = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    "TwoPatterns",
    "FordA",
    # "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]
# NUmber of runs per experiment
NUM_RUNS_PER = 1
# Controls
RUN_STAGE1 = True
RUN_STAGE2 = True

SEED = 1
# Epochs:
STAGE1_EPOCHS = 1000  # 1000
STAGE2_EPOCHS = 1000

STAGE1_AUGS = ["window_warp", "amplitude_resize"]


# Main experiment function
def run_experiments():
    # Set manual seed
    torch.manual_seed(SEED)

    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)
    # Set max epochs for each stage
    config["trainer_params"]["max_epochs"]["stage1"] = STAGE1_EPOCHS
    config["trainer_params"]["max_epochs"]["stage2"] = STAGE2_EPOCHS
    # Only reconstruct original view:

    config["augmentations"]["time_augs"] = STAGE1_AUGS

    config["seed"] = SEED
    config["id"] = generate_short_id(
        length=6
    )  # all models in the experiment will use this id.

    batch_size_stage1 = config["dataset"]["batch_sizes"]["stage1"]
    batch_size_stage2 = config["dataset"]["batch_sizes"]["stage2"]

    # List of experiments to run
    experiments = []

    if RUN_STAGE1:
        experiments += [
            # Stage 1
            {
                "stage": 1,
                "stage1_exp": exp,
                "augmented_data": (exp != ""),
                "orthogonal_reg_weight": ortho_reg,
                "project_name": STAGE1_PROJECT_NAME,
                "epochs": STAGE1_EPOCHS,
                "train_fn": train_vqvae if exp == "" else train_ssl_vqvae,
                "full_embed": False,
            }
            for ortho_reg in [0, 10]
            for exp in STAGE1_EXPS
        ]

    if RUN_STAGE2:
        experiments += [
            # Stage 2
            {
                "stage": 2,
                "stage1_exp": exp,
                "augmented_data": False,
                "orthogonal_reg_weight": ortho_reg,
                "project_name": STAGE2_PROJECT_NAME,
                "epochs": STAGE2_EPOCHS,
                "train_fn": train_maskgit,
                "full_embed": (exp != ""),
            }
            for ortho_reg in [0, 10]
            for exp in STAGE1_EXPS
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
            c["SSL"][f"stage1_method"] = experiment["stage1_exp"]
            c["VQVAE"]["orthogonal_reg_weight"] = experiment["orthogonal_reg_weight"]

            for run in range(NUM_RUNS_PER):
                # Wandb run name:
                run_name = experiment_name(experiment, SEED)

                # Set correct data loader
                if experiment["stage"] == 1:
                    train_data_loader = (
                        train_data_loader_stage1_aug
                        if experiment["augmented_data"]
                        else train_data_loader_stage1
                    )
                    c["trainer_params"]["max_epochs"]["stage1"] = experiment["epochs"]
                else:
                    train_data_loader = train_data_loader_stage2
                    c["trainer_params"]["max_epochs"]["stage2"] = experiment["epochs"]

                experiment["train_fn"](
                    config=c,
                    train_data_loader=train_data_loader,
                    test_data_loader=test_data_loader,
                    do_validate=True,
                    gpu_device_idx=0,
                    wandb_run_name=f"{run_name}-{dataset}",
                    wandb_project_name=experiment["project_name"],
                    torch_seed=SEED,
                    full_embed=experiment["full_embed"],
                )


if __name__ == "__main__":
    run_experiments()
