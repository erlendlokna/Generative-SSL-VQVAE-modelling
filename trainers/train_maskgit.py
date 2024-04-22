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
from experiments.exp_full_embed_maskgit import ExpFullEmbedMaskGIT
from evaluation.model_eval import Evaluation
import matplotlib.pyplot as plt

# from evaluation.evaluation import Evaluation
from utils import (
    get_root_dir,
    load_yaml_param_settings,
    save_model,
    save_codebook,
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


def train_maskgit(
    config: dict,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    gpu_device_idx: int,
    do_validate: bool,
    wandb_project_name: str = "SSL_VQVAE-stage2",
    wandb_run_name="",
    torch_seed=0,
    full_embed=False,
    finetune_codebook=False,
):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    torch.manual_seed(torch_seed)

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]

    # initiate model:
    if full_embed:
        train_exp = ExpFullEmbedMaskGIT(
            input_length=input_length,
            config=config,
            n_train_samples=len(train_data_loader.dataset),
            n_classes=n_classes,
            finetune_codebook=finetune_codebook,
            load_finetuned_codebook=False,
            device=gpu_device_idx,
        )
        exp_name = "fullembed-maskgit"
        exp_name += "-finetuned" if finetune_codebook else ""
    else:
        train_exp = ExpMaskGIT(
            input_length, config, len(train_data_loader.dataset), n_classes
        )
        exp_name = "maskgit"

    wandb_logger = WandbLogger(
        project=wandb_project_name, name=wandb_run_name, config=config
    )

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
        {model_filename(config, exp_name): train_exp.maskgit},
        id=config["dataset"]["dataset_name"],
    )
    # Save codebook if it is finetuned
    if finetune_codebook and full_embed:
        save_codebook(
            {model_filename(config, "finetuned_codebook"): train_exp.maskgit.cb_stage1},
            id=config["dataset"]["dataset_name"],
        )

    print("evaluating...")
    print("FID, IS, PCA, TSNE")
    dataset_name = config["dataset"]["dataset_name"]
    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    evaluation = Evaluation(
        subset_dataset_name=dataset_name,
        gpu_device_index=gpu_device_idx,
        config=config,
        batch_size=config["dataset"]["batch_sizes"]["stage2"],
    )
    if full_embed:
        x_gen = evaluation.sampleFullEmbedMaskGit(
            max(
                evaluation.X_test.shape[0],
                config["dataset"]["batch_sizes"]["stage2"],
            ),
            input_length,
            n_classes,
            "unconditional",
            device=gpu_device_idx,
            load_finetuned_codebook=finetune_codebook,
            # Load the finetuned codebook
        )
    else:
        x_gen = evaluation.sampleMaskGit(
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

    print("Aggregating iterative decodings")
    agg_samples, agg_sel_probs, agg_entropy, agg_sel_entropy = (
        evaluation.aggregate_statistics(train_exp.maskgit, num_iterations=100)
    )

    evaluation.log_iterative_decoding_statistics(
        agg_sel_probs, agg_entropy, agg_sel_entropy
    )

    # vocabulary and variety
    num_tokens_idx = train_exp.maskgit.mask_token_ids
    evaluation.log_coverage_and_variety(agg_samples, num_tokens_idx)

    co_occurence = evaluation.co_occurence_matrix(agg_samples, num_tokens_idx)
    evaluation.log_co_occurence(co_occurence)
    # Calculate probabilities
    token_prob, joint_prob, conditional_prob = evaluation.calculate_probabilities(
        co_occurence
    )

    evaluation.log_prior_token_ratios(token_prob)

    evaluation.log_conditional_probs(conditional_prob)

    pmi = evaluation.calculate_pmi(token_prob, joint_prob)

    evaluation.log_pmi(pmi)

    evaluation.log_pmi_vs_usage(pmi, token_prob)

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
    train_maskgit(
        config,
        train_data_loader,
        test_data_loader,
        args.gpu_device_idx,
        do_validate=True,
    )
