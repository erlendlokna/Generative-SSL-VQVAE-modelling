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
STAGE1_PROJECT_NAME = "S1-Master-Run"
STAGE2_PROJECT_NAME = "S2-Master-Run"

# Datasets to run experiments on
UCR_SUBSET = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    # "TwoPatterns",
    "FordA",
    "UWaveGestureLibraryAll",
    "FordB",
    "ShapesAll",
]
# NUmber of runs per experiment
NUM_RUNS_PER = 1  # Will overwrite models in saved_models. Recomennded to set to 1.
# Controls
RUN_STAGE1 = True
RUN_STAGE2 = True

SEEDS = [4]

# Epochs:
STAGE1_EPOCHS = 1000
STAGE2_EPOCHS = 1000

STAGE1_AUGS = ["window_warp", "amplitude_resize"]
AUG_RECON_RATE = 0.1

# Stage 1 SSL methods to run
SSL_METHODS = [
    "",
    "vibcreg",
    "barlowtwins",
]  # empty string means regular VQVAE / no SSL

FINETUNE_CODEBOOK = True
INCLUDE_DECORRELATION = True


def generate_experiments():
    experiments = []
    orhogonal_reg_weights = [0, 10] if INCLUDE_DECORRELATION else [0]

    if RUN_STAGE1:
        experiments += [
            {
                "stage": 1,
                "ssl_method": ssl_method,
                "augmented_data": (ssl_method != ""),
                "orthogonal_reg_weight": ortho_reg,
                "project_name": STAGE1_PROJECT_NAME,
                "epochs": STAGE1_EPOCHS,
                "train_fn": train_vqvae if ssl_method == "" else train_ssl_vqvae,
                # train_fn is train_vqvae if no SSL method, else train_ssl_vqvae
                "full_embed": False,  # Stage 1 does not use full embed
                "finetune_codebook": False,  # Stage 1 does not finetune codebook
            }
            for ortho_reg in orhogonal_reg_weights
            for ssl_method in SSL_METHODS
        ]

    if RUN_STAGE2:
        experiments += [
            {
                "stage": 2,
                "ssl_method": ssl_method,
                "augmented_data": False,
                "orthogonal_reg_weight": ortho_reg,
                "project_name": STAGE2_PROJECT_NAME,
                "epochs": STAGE2_EPOCHS,
                "train_fn": train_maskgit,
                "full_embed": (ssl_method != ""),  # full embed only for SSL methods
                "finetune_codebook": (ssl_method != "") and FINETUNE_CODEBOOK,
                # finetune codebook only for SSL methods
            }
            for ortho_reg in orhogonal_reg_weights
            for ssl_method in SSL_METHODS
        ]
    return experiments


def build_data_pipelines(config):
    print(
        "Generating data pipelines for {}...".format(config["dataset"]["dataset_name"])
    )
    batch_size_stage1 = config["dataset"]["batch_sizes"]["stage1"]
    batch_size_stage2 = config["dataset"]["batch_sizes"]["stage2"]
    # Build data pipelines
    dataset_importer = UCRDatasetImporter(**config["dataset"])
    train_data_loader_stage1 = build_data_pipeline(
        batch_size_stage1, dataset_importer, config, augment=False, kind="train"
    )
    train_data_loader_stage1_aug = build_data_pipeline(
        batch_size_stage1, dataset_importer, config, augment=True, kind="train"
    )
    train_data_loader_stage2 = build_data_pipeline(
        batch_size_stage2, dataset_importer, config, augment=False, kind="train"
    )
    test_data_loader = build_data_pipeline(
        batch_size_stage1, dataset_importer, config, augment=False, kind="test"
    )  # Same test dataloader for both stages

    return (
        train_data_loader_stage1,
        train_data_loader_stage1_aug,
        train_data_loader_stage2,
        test_data_loader,
    )


# Main experiment function
def run_experiments(seed):
    # Set manual seed
    torch.manual_seed(seed)
    # load config
    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)
    c = config.copy()

    # Set max epochs for each stage
    c["trainer_params"]["max_epochs"]["stage1"] = STAGE1_EPOCHS
    c["trainer_params"]["max_epochs"]["stage2"] = STAGE2_EPOCHS
    c["augmentations"]["time_augs"] = STAGE1_AUGS
    c["VQVAE"]["aug_recon_rate"] = AUG_RECON_RATE
    c["seed"] = seed
    c["ID"] = generate_short_id(length=6)
    # all models in the experiment will use this id.

    experiments = generate_experiments()  # Generate experiments to run.

    print("Experiments to run:")
    for i, exp in enumerate(experiments):
        print(f"{i+1}. {exp}\n")

    for dataset in UCR_SUBSET:
        c["dataset"]["dataset_name"] = dataset

        # Build data pipelines
        (
            train_data_loader_stage1,
            train_data_loader_stage1_aug,
            train_data_loader_stage2,
            test_data_loader,
        ) = build_data_pipelines(
            c
        )  # Will be work on the torch.manual(seed)

        # Running experiments:
        for experiment in experiments:
            # Only configure stage 1 method:
            c["SSL"][f"stage1_method"] = experiment["ssl_method"]
            c["VQVAE"]["orthogonal_reg_weight"] = experiment["orthogonal_reg_weight"]

            for run in range(NUM_RUNS_PER):
                # Wandb run name:
                run_name = experiment_name(experiment, seed, c["ID"])
                run_name += "finetune" if experiment["finetune_codebook"] else ""
                # Set correct data loader
                if experiment["stage"] == 1:
                    train_data_loader = (
                        train_data_loader_stage1_aug
                        if experiment["augmented_data"]
                        else train_data_loader_stage1
                    )
                else:
                    train_data_loader = train_data_loader_stage2

                # experiment trainer:
                experiment["train_fn"](
                    # Stage 1 and 2:
                    config=c,
                    train_data_loader=train_data_loader,
                    test_data_loader=test_data_loader,
                    do_validate=True,
                    gpu_device_idx=0,
                    wandb_run_name=f"{run_name}-{dataset}",
                    wandb_project_name=experiment["project_name"],
                    torch_seed=seed,
                    # Stage 2 using SSL:
                    full_embed=experiment["full_embed"],
                    finetune_codebook=experiment["finetune_codebook"],
                )


if __name__ == "__main__":
    for seed in SEEDS:
        run_experiments(seed)
