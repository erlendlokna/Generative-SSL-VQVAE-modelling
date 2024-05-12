"""
CAS (= TSTR):
- Train on the Synthetic samples, and
- Test on the Real samples.

[1] Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020).
"""

from argparse import ArgumentParser

import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, get_root_dir
from evaluation.cas import SyntheticDataset

from experiments.exp_fcn import ExpFCN


def train_FCN(
    train_data_loader,
    real_test_data_loader,
    dataset_importer,
    config,
    config_cas,
    wandb_project,
    wandb_run_name,
):

    # fit
    train_exp = ExpFCN(
        config_cas,
        len(train_data_loader.dataset),
        len(np.unique(train_data_loader.dataset.Y_gen)),
    )
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_run_name,  # config["dataset"]["dataset_name"],
        config=config_cas | config,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        devices=config_cas["trainer_params"]["gpus"],
        accelerator="gpu",
        max_epochs=config_cas["trainer_params"]["max_epochs"],
    )
    trainer.fit(
        train_exp,
        train_dataloaders=train_data_loader,
        val_dataloaders=real_test_data_loader,
    )

    # visual comp btn real and synthetic
    fig, axes = plt.subplots(2, 1, figsize=(4, 4))
    axes = axes.flatten()
    n_samples = min(dataset_importer.X_train.shape[0], 200)
    ind0 = np.random.randint(0, dataset_importer.X_train.shape[0], size=n_samples)
    ind1 = np.random.randint(
        0, train_data_loader.dataset.X_gen.shape[0], size=n_samples
    )

    X_train = dataset_importer.X_train[ind0]  # (n_samples len)
    Y_train = dataset_importer.Y_train[ind0]  # (n_samples 1)
    X_gen = train_data_loader.dataset.X_gen.squeeze()[ind1]  # (n_samples len)
    Y_gen = train_data_loader.dataset.Y_gen.squeeze()[ind1]  # (n_samples 1)

    axes[0].plot(X_train.T, alpha=0.1)
    axes[1].plot(X_gen.T, alpha=0.1)
    plt.tight_layout()
    wandb.log({"real vs synthetic": wandb.Image(plt)})

    # finish wandb
    wandb.finish()
