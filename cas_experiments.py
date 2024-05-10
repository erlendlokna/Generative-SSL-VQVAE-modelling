from trainers.train_FCN import train_FCN

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
    generate_short_id,
    experiment_name,
    search_relevant_ids,
)
from torch.utils.data import DataLoader
from evaluation.cas import SyntheticDataset, Sampler

from trainers.train_vqvae import train_vqvae
from trainers.train_ssl_vqvae import train_ssl_vqvae
from trainers.train_maskgit import train_maskgit
import torch


# Wandb logging information
CAS_PROJECT_NAME = "Master-CAS-Run"

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
    # "ChlorineConcentration",
    "ShapesAll",
]

# Stage 1 SSL methods to run
SSL_METHODS = [
    "",
    "vibcreg",
    "barlowtwins",
]  # empty string means regular VQVAE / no SSL


SEEDS = [2]


def generate_experiments():
    experiments = []
    experiments += [
        {
            "ssl_method": ssl_method,
            "project_name": CAS_PROJECT_NAME,
            "train_fn": train_FCN,
        }
        for ssl_method in SSL_METHODS
    ]
    return experiments


def build_data_pipelines(config, config_cas, synthetic=False):
    dataset_importer = UCRDatasetImporter(**config["dataset"])
    batch_size = config["dataset"]["batch_sizes"]["stage2"]
    real_train_data_loader, real_test_data_loader = [
        build_data_pipeline(batch_size, dataset_importer, config, kind)
        for kind in ["train", "test"]
    ]

    ssl_stage1 = config["SSL"]["stage1_method"] != ""

    if not synthetic:
        return real_train_data_loader, real_test_data_loader, dataset_importer

    train_data_loader = DataLoader(
        SyntheticDataset(
            real_train_data_loader,
            1000,
            config,
            config_cas["dataset"]["batch_size"],
            config_cas["trainer_params"]["gpus"][0],
            ssl_stage1=ssl_stage1,
        ),
        batch_size=config_cas["dataset"]["batch_size"],
        num_workers=config_cas["dataset"]["num_workers"],
        shuffle=True,
        drop_last=False,
    )
    return train_data_loader


def run_cas_experiments(seed):
    torch.manual_seed(seed)

    config_dir = get_root_dir().joinpath("configs", "config.yaml")

    config = load_yaml_param_settings(
        get_root_dir().joinpath("configs", "config.yaml")
    )  # load regular config
    config_cas = load_yaml_param_settings(
        get_root_dir().joinpath("configs", "config_cas.yaml")
    )  # load cas config

    c = config.copy()
    c_cas = config_cas.copy()

    c["seed"] = seed

    experiments = generate_experiments()

    for dataset in UCR_SUBSET:
        c["dataset"]["dataset_name"] = dataset

        (
            real_train_data_loader,
            real_test_data_loader,
            dataset_importer,
        ) = build_data_pipelines(c, c_cas, synthetic=False)

        for exp in experiments:

            c["SSL"]["stage1_method"] = exp["ssl_method"]

            # IDs to runs on dataset with ssl method
            relevant_ids = search_relevant_ids(c)
            print(f"Running {exp['ssl_method']} on {dataset} with ids {relevant_ids}")
            for id in relevant_ids:
                c["ID"] = id

                synthetic_data_loader = build_data_pipelines(c, c_cas, synthetic=True)
                method_name = f"{exp['ssl_method']}-" if exp["ssl_method"] else ""
                run_name = f"CAS-{method_name}seed{seed}-{c['ID']}-{dataset}"

                exp["train_fn"](
                    synthetic_data_loader,
                    real_test_data_loader,
                    dataset_importer,
                    c,
                    c_cas,
                    exp["project_name"],
                    wandb_run_name=run_name,
                )


if __name__ == "__main__":
    for seed in SEEDS:
        run_cas_experiments(seed)
