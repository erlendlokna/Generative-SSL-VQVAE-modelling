from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
)
import torch

from trainers.train_vqvae import train_vqvae
from trainers.train_ssl_vqvae import train_ssl_vqvae

UCR_SUBSET = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    # "TwoPatterns",
    # "FordA",
    "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]

NUM_RUNS_PER = 5

STAGE1_EPOCHS = 500

STAGE1_METHODS = ["", "vibcreg"]
SSL_WEIGHTS = {"barlowtwins": 1.0, "vibcreg": 0.01, "vicreg": 0.01, "": 0}


def run_experiments():
    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)

    config["trainer_params"]["max_epochs"]["stage1"] = STAGE1_EPOCHS

    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    project_name = "codebook analysis"

    for dataset in UCR_SUBSET:
        c = config.copy()
        c["dataset"]["dataset_name"] = dataset

        dataset_importer = UCRDatasetImporter(**config["dataset"])

        train_data_loader_no_aug = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="train"
        )
        train_data_loader_aug = build_data_pipeline(
            batch_size, dataset_importer, config, augment=True, kind="train"
        )
        test_data_loader = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="test"
        )

        # for run in range(NUM_RUNS_PER):
        #    train_vqvae(
        #        c,
        #        train_data_loader_no_aug,
        #        test_data_loader,
        #        gpu_device_idx=0,
        #        do_validate=True,
        #        wandb_project_name=project_name,
        #        wandb_run_name=f"VQVAE-run{run}",
        #    )

        # for run in range(NUM_RUNS_PER):
        #    c["VQVAE"]["decorrelate_codebook"] = False
        #    train_ssl_vqvae(
        #        c,
        #        train_data_loader_aug,
        #        test_data_loader,
        #        gpu_device_idx=0,
        #        do_validate=True,
        #        wandb_project_name=project_name,
        #        wandb_run_name=f"SSLVQVAE_run{run}",
        #    )

        for run in range(NUM_RUNS_PER):
            c["VQVAE"]["decorr_codebook"] = True
            c["VQVAE"]["decorr_weight_schedule"] = False
            train_ssl_vqvae(
                c,
                train_data_loader_aug,
                test_data_loader,
                gpu_device_idx=0,
                do_validate=True,
                wandb_project_name=project_name,
                wandb_run_name=f"SSLVQVAE_decorr_run{run}",
            )
        for run in range(NUM_RUNS_PER):
            c["VQVAE"]["decorr_codebook"] = True
            c["VQVAE"]["decorr_weight_schedule"] = True
            c["VQVAE"]["decorr_weight_schedule_p"] = 2
            train_ssl_vqvae(
                c,
                train_data_loader_aug,
                test_data_loader,
                gpu_device_idx=0,
                do_validate=True,
                wandb_project_name=project_name,
                wandb_run_name=f"SSLVQVAE_decorr_ws2_run{run}",
            )
        for run in range(NUM_RUNS_PER):
            c["VQVAE"]["decorr_codebook"] = True
            c["VQVAE"]["decorr_weight_schedule"] = True
            c["VQVAE"]["decorr_weight_schedule_p"] = 10
            train_ssl_vqvae(
                c,
                train_data_loader_aug,
                test_data_loader,
                gpu_device_idx=0,
                do_validate=True,
                wandb_project_name=project_name,
                wandb_run_name=f"SSLVQVAE_decorr_ws5_run{run}",
            )


if __name__ == "__main__":
    run_experiments()
