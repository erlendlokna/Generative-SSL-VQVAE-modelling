import copy
from argparse import ArgumentParser

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import UCRDatasetImporter
from experiments.exp_maskgit import ExpMaskGIT
from experiments.exp_mage import ExpMAGE
from evaluation.model_eval import Evaluation

# from evaluation.evaluation import Evaluation
from utils import (
    get_root_dir,
    load_yaml_param_settings,
    save_model,
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
    parser.add_argument(
        "--dataset_names", nargs="+", help="e.g., Adiac Wafer Crop`.", default=""
    )
    parser.add_argument("--gpu_device_idx", default=0, type=int)
    return parser.parse_args()


def train_mage(
    ssl_stage2: bool,
    config: dict,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    gpu_device_idx: int,
    do_validate: bool,
    wandb_project_name: str = "SSL_VQVAE-stage2",
    wandb_run_name="",
):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = "SSL_VQVAE-stage2"

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]

    # initiate model:
    train_exp = ExpMAGE(
        input_length,
        config,
        len(train_data_loader.dataset),
        n_classes,
        train_data_loader,
        test_data_loader,
    )

    wandb_logger = WandbLogger(project=project_name, name=None, config=config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        max_epochs=config["trainer_params"]["max_epochs"]["stage2"],
        devices=[
            gpu_device_idx,
        ],
        accelerator="gpu",
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

    print("saving the model...")
    save_model(
        {model_filename(config, "MAGE"): train_exp.MAGE},
        id=config["dataset"]["dataset_name"],
    )

    # test
    print("evaluating...")
    dataset_name = config["dataset"]["dataset_name"]
    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    evaluation = Evaluation(
        subset_dataset_name=dataset_name,
        gpu_device_index=gpu_device_idx,
        config=config,
        batch_size=config["dataset"]["batch_sizes"]["stage2"],
    )

    x_gen = evaluation.sampleMAGE(
        max(
            evaluation.X_test.shape[0],
            config["dataset"]["batch_sizes"]["stage2"],
        ),
        input_length,
        n_classes,
        "unconditional",
    )

    z_test, z_gen = evaluation.compute_z(x_gen)
    fid, (z_test, z_gen) = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({"FID": fid, "IS_mean": IS_mean, "IS_std": IS_std})

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)

    wandb.finish()


if __name__ == "__main__":
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    dataset_importer = UCRDatasetImporter(**config["dataset"])
    batch_size = config["dataset"]["batch_sizes"]["stage2"]

    train_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="train"
    )
    test_data_loader = build_data_pipeline(
        batch_size, dataset_importer, config, augment=False, kind="test"
    )

    # train
    ssl_stage2 = True
    print(f"starting training stage 2 using {'MAGE' if ssl_stage2 else'MaskGIT'}")
    train_mage(
        ssl_stage2,
        config,
        train_data_loader,
        test_data_loader,
        args.gpu_device_idx,
        do_validate=True,
    )
