import copy
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_vqvae import Exp_VQVAE

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, save_model, get_root_dir, model_filename
import torch
import time

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


def train_vqvae(
    config: dict,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    do_validate: bool,
    gpu_device_idx: int,
    wandb_project_name: str = "SSL_VQVAE-stage1",
    wandb_run_name="",
    disable_wandb=False,
    torch_seed=0,
):
    """
    Trainer for VQVAE or SSL-VQVAE model based on the `do_validate` and `SSL` parameters.
    """

    torch.manual_seed(torch_seed)

    input_length = train_data_loader.dataset.X.shape[-1]

    train_exp = Exp_VQVAE(
        input_length,
        config=config,
        n_train_samples=len(train_data_loader.dataset),
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
    )
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=wandb_run_name,
        config=config,
        mode="disabled" if disable_wandb else "online",
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
    start_time = time.time()
    # Your training code here
    trainer.fit(
        train_exp,
        train_dataloaders=train_data_loader,
        val_dataloaders=test_data_loader if do_validate else None,
    )
    end_time = time.time()

    wandb.log({"training_time": end_time - start_time})

    # additional log
    n_trainable_params = sum(
        p.numel() for p in train_exp.parameters() if p.requires_grad
    )
    wandb.log({"n_trainable_params:": n_trainable_params})

    # test
    wandb.finish()

    # Save the models
    print("saving models...")
    save_model(
        {
            model_filename(config, "encoder"): train_exp.encoder,
            model_filename(config, "decoder"): train_exp.decoder,
            model_filename(config, "vqmodel"): train_exp.vq_model,
        },
        id=config["dataset"]["dataset_name"],
    )


if __name__ == "__main__":
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    dataset_importer = UCRDatasetImporter(**config["dataset"])
    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    train_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="train"
    )
    test_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="test"
    )

    # Train VQVAE without validation
    print("Starting training...")
    train_vqvae(
        config=config,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        gpu_device_idx=args.gpu_device_idx,
        do_validate=True,
    )
