import copy
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_ssl_vqvae import Exp_SSL_VQVAE
from experiments.exp_vqvae import Exp_VQVAE

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, save_model, get_root_dir
import torch


torch.set_float32_matmul_precision("medium")


def load_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config data  file.",
        default=get_root_dir().joinpath("configs", "config.yaml"),
    )
    parser.add_argument(
        "--dataset_names", nargs="+", help="e.g., Adiac Wafer Crop`.", default=""
    )
    parser.add_argument("--gpu_device_idx", default=0, type=int)
    return parser.parse_args()


def train_stage1(
    SSL: bool,
    config: dict,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    do_validate: bool,
    gpu_device_idx: int,
    wandb_project_case_idx: str = "",
    wandb_run_name="",
):
    """
    Trainer for VQVAE or SSL-VQVAE model based on the `do_validate` and `SSL` parameters.
    """
    project_name = "SSL_VQVAE-stage1"

    if wandb_project_case_idx != "":
        project_name += f"-{wandb_project_case_idx}"

    input_length = train_data_loader.dataset.X.shape[-1]

    if SSL:
        train_exp = Exp_SSL_VQVAE(
            input_length,
            config=config,
            n_train_samples=len(train_data_loader.dataset),
        )
    else:
        train_exp = Exp_VQVAE(
            input_length,
            config=config,
            n_train_samples=len(train_data_loader.dataset),
        )

    wandb_logger = WandbLogger(
        project=project_name,
        name=wandb_run_name,
        config=config,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        max_epochs=(config["trainer_params"]["max_epochs"]["stage1"]),
        devices=[
            (
                gpu_device_idx
                if torch.cuda.is_available() and gpu_device_idx >= 0
                else "cpu"
            )
        ],
        accelerator=(
            "gpu" if torch.cuda.is_available() and gpu_device_idx >= 0 else "cpu"
        ),
        check_val_every_n_epoch=20,
    )

    trainer.fit(
        train_exp,
        train_dataloaders=train_data_loader,
        val_dataloaders=test_data_loader if do_validate else None,
    )

    # additional log
    n_trainable_params = sum(
        p.numel() for p in train_exp.parameters() if p.requires_grad
    )
    wandb.log({"n_trainable_params:": n_trainable_params})

    # test
    wandb.finish()

    # Save the models
    prefix = (
        ""
        if not SSL
        else f"{config['SSL']['method_choice']}_{config['SSL']['weighting']}_"
    )

    save_model(
        {
            f"{prefix}encoder": train_exp.encoder,
            f"{prefix}decoder": train_exp.decoder,
            f"{prefix}vq_model": train_exp.vq_model,
        },
        id=config["dataset"]["dataset_name"],
    )


if __name__ == "__main__":
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    use_ssl = config["VQVAE"]["SSL"]

    dataset_importer = UCRDatasetImporter(**config["dataset"])
    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    train_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=use_ssl, kind="train"
    )
    test_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="test"
    )

    # Train VQVAE without validation
    train_stage1(
        SSL=use_ssl,
        config=config,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        gpu_device_idx=args.gpu_device_idx,
        do_validate=True,
    )
