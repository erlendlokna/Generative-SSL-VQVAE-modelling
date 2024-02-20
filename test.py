import copy
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.stage2.mage import MAGE
from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, save_model, get_root_dir
import torch
import numpy as np


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


def test():
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

    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))

    mage = MAGE(input_length, **config["MAGE"], config=config, n_classes=n_classes)

    for batch in train_data_loader:
        x, y = batch
        logits, summary, target = mage(x, y, return_summaries=True)
        break

    print("logits1.shape", logits[0].shape)
    print("summary1.shape", summary[0].shape)
    print("target1.shape", target[0].shape)


if __name__ == "__main__":
    test()
