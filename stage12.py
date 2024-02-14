import copy
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_nc_vqvae import Exp_SSL_VQVAE
from experiments.exp_vqvae import Exp_VQVAE

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, save_model, get_root_dir
import torch


from stage1 import train_stage1
from stage2 import train_stage2


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


ucr_subset = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    "ECG5000",
    # "TwoPatterns",
    # "FordA",
    # "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]


stage1_ssl_methods = ["barlowtwins"]
stage2_ssl_methods = ["barlowtwins"]


def run_experiment():
    args = load_args()
    config = load_yaml_param_settings(args.config)

    assert (
        stage1_ssl_methods.sort() == stage2_ssl_methods.sort()
    ), "SSL methods should be the same for stage1 and stage2"

    # Load the datasets
    for dataset_name in ucr_subset:
        # configure config and datasets
        c = copy.deepcopy(config)
        c["dataset"]["dataset_name"] = dataset_name
        c["SSL"]["stage1_ssl"] = ""  # No SSL

        dataset_importer = UCRDatasetImporter(**c["dataset"])

        batch_size_stage1 = c["dataset"]["batch_sizes"]["stage1"]
        batch_size_stage2 = c["dataset"]["batch_sizes"]["stage2"]

        test_data_loader = lambda batch_size: build_data_pipeline(
            batch_size, dataset_importer, c, augment=False, kind="test"
        )

        train_data_loader = lambda batch_size, augment: build_data_pipeline(
            batch_size, dataset_importer, c, augment=augment, kind="train"
        )

        # start training
        print("Training stage 1 without SSL..", dataset_name)
        # stage 1 without SSL
        train_stage1(
            config=c,
            train_data_loader=train_data_loader(batch_size_stage1, augment=False),
            test_data_loader=test_data_loader(batch_size_stage1),
            gpu_device_idx=args.gpu_device_idx,
            do_validate=True,
        )

        print("Training stage 2 without SSL..")
        train_stage2(
            config=c,
            train_data_loader=train_data_loader(batch_size_stage2, False),
            test_data_loader=test_data_loader(batch_size_stage2),
            do_validate=True,
            gpu_device_idx=args.gpu_device_idx,
        )

        for stage1_ssl_method in stage1_ssl_methods:
            c["SSL"]["stage1_method"] = stage1_ssl_method

        for ssl_method in ssl_methods:
            c["VQVAE"]["ssl_method"] = ssl_method  # Adding SSL method

            train_stage1(
                config=c,
                train_data_loader=train_data_loader(
                    batch_size_stage1, augment_stage1(c)
                ),
                test_data_loader=test_data_loader(batch_size_stage1),
                gpu_device_idx=args.gpu_device_idx,
                do_validate=True,
            )

            print("Training stage 2...")
            train_stage2(
                config=c,
                train_data_loader=train_data_loader(batch_size_stage2, augment_stage2),
                test_data_loader=test_data_loader(batch_size_stage2),
                do_validate=True,
                gpu_device_idx=args.gpu_device_idx,
            )


if __name__ == "__main__":
    main()
