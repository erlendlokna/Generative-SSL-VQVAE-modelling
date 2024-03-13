from preprocessing.data_pipeline import build_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import UCRDatasetImporter
from argparse import ArgumentParser

from trainers.train_vqvae import train_vqvae
from trainers.train_ssl_vqvae import train_ssl_vqvae
from trainers.train_mage import train_mage
from trainers.train_byol_maskgit import train_byol_maskgit
from trainers.train_maskgit import train_maskgit
import torch

from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
)


def load_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config data  file.",
        default=get_root_dir().joinpath("configs", "config.yaml"),
    )

    parser.add_argument("--dataset_name", type=str, default="ElectricDevices")

    parser.add_argument("--ssl_stage1", default="", type=str)
    parser.add_argument("--ssl_stage2", default="", type=str)

    parser.add_argument("--model", type=str, default="vqvae")

    parser.add_argument("--gpu_device_idx", default=0, type=int)

    parser.add_argument("--epochs_stage1", default=2000, type=int)
    parser.add_argument("--epochs_stage2", default=10000, type=int)

    parser.add_argument("--disable_wandb", default=False, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)

    args = load_args()
    config = load_yaml_param_settings(args.config)
    # config["dataset"]["name"] = "FordA"
    # config["SSL"]["stage1_method"] = args.ssl_stage1
    # config["SSL"]["stage2_method"] = args.ssl_stage2
    # config["trainer_params"]["max_epochs"]["stage1"] = 1000  # args.epochs_stage1
    # config["trainer_params"]["max_epochs"]["stage2"] = args.epochs_stage2
    disable_wandb = args.disable_wandb

    dataset_importer = UCRDatasetImporter(**config["dataset"])
    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    train_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="train"
    )

    train_data_loader_aug = build_data_pipeline(
        batch_size, dataset_importer, config, augment=True, kind="train"
    )

    test_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="test"
    )

    wandb_project = "codebook analysis"

    if args.model == "vqvae":
        train_vqvae(
            config,
            train_data_loader,
            test_data_loader,
            do_validate=True,
            gpu_device_idx=args.gpu_device_idx,
            disable_wandb=disable_wandb,
            wandb_project_name=wandb_project,
            torch_seed=0,
        )
    elif args.model == "sslvqvae":
        train_ssl_vqvae(
            config,
            train_data_loader_aug,
            test_data_loader,
            do_validate=True,
            gpu_device_idx=args.gpu_device_idx,
            disable_wandb=disable_wandb,
            wandb_project_name=wandb_project,
            torch_seed=0,
        )

    elif args.model == "byolmaskgit":
        train_byol_maskgit(
            config,
            train_data_loader,
            test_data_loader,
            do_validate=True,
            gpu_device_idx=args.gpu_device_idx,
        )
    elif args.model == "maskgit":
        train_maskgit(
            config,
            train_data_loader,
            test_data_loader,
            do_validate=True,
            gpu_device_idx=args.gpu_device_idx,
            torch_seed=0,
        )
    else:
        raise ValueError(
            f"Unknown model name: {args.model_name}. Please choose one of (mage, vqvae, maskgit, sslmaskgit, sslvqvae). Exiting..."
        )
