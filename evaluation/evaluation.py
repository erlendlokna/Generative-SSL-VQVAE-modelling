"""
FID, IS, JS divergence.
"""

import numpy as np
import os
import copy
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.maskgit import MaskGIT
from preprocessing.preprocess_ucr import UCRDatasetImporter
from models.sample import unconditional_sample, conditional_sample
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score
from utils import (
    time_to_timefreq,
    timefreq_to_time,
    ssl_config_filename,
)
from preprocessing.data_pipeline import build_data_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


class ExpEvaluation(object):
    def __init__(self, config, device):
        self.device = device
        self.dataset_name = config["dataset"]["dataset_name"]
        self.batch_size = config["dataset"]["batch_sizes"]["stage2"]

        self.fcn = load_pretrained_FCN(self.dataset_name).to(device)
        self.fcn.eval()

        dataset_importer = UCRDatasetImporter(**config["dataset"])

        self.X_train = dataset_importer.X_train[:, None, :]
        self.X_test = dataset_importer.X_test[:, None, :]

        self.Y_train = dataset_importer.Y_train
        self.Y_test = dataset_importer.Y_test

        self.config = config

    @torch.no_grad()
    def sample_eval(self, model, n_samples=None, device: torch.device = None):
        if n_samples is None:
            n_samples = self.X_test.shape[0]

        model.eval()

        uncond_sample = self.sample(model, n_samples, "unconditional", device)

        # Convert generated samples to feature vectors
        z_test, z_gen = self.compute_z(uncond_sample)

        # Calculate FID score between real and generated samples
        fid_score = calculate_fid(z_test, z_gen)

        return {"fid_score": fid_score}

    def downstream_summary_eval(self, MAGE, device):

        MAGE.eval()

        # Convert X_train and y_train to PyTorch tensors and move them to the specified device
        X_train = torch.from_numpy(self.X_train).float()
        X_test = torch.from_numpy(self.X_test).float()

        # Use MAGE.summarize on X_train and y_train
        summary_train = MAGE.summarize(X_train.to(device)).cpu()
        summary_test = MAGE.summarize(X_test.to(device)).cpu()

        Y_train = self.Y_train.flatten()
        Y_test = self.Y_test.flatten()

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(summary_train)
        summary_train = scaler.transform(summary_train)
        summary_test = scaler.transform(summary_test)

        knn = KNeighborsClassifier()
        knn.fit(summary_train, Y_train)
        preds = knn.predict(summary_test)

        return {"knn_accuracy": metrics.accuracy_score(Y_test, preds)}

    @torch.no_grad()
    def sample(
        self,
        model,
        n_samples: int,
        kind: str,
        device: torch.device,
        class_index: int = -1,
    ):

        assert kind in ["unconditional", "conditional"]

        # sampling
        if kind == "unconditional":
            x_new = unconditional_sample(
                model,
                n_samples,
                device,
                batch_size=self.batch_size,
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new = conditional_sample(
                model,
                n_samples,
                device,
                class_index,
                self.batch_size,
            )  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        return x_new

    @torch.no_grad()
    def compute_z(self, X_gen: torch.Tensor):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test` and `X_gen`
        z_test, z_gen = [], []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            z_t = (
                self.fcn(
                    torch.from_numpy(self.X_test[s]).float().to(self.device),
                    return_feature_vector=True,
                )
                .cpu()
                .detach()
                .numpy()
            )
            z_g = (
                self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True)
                .cpu()
                .detach()
                .numpy()
            )

            z_test.append(z_t)
            z_gen.append(z_g)
        z_test, z_gen = np.concatenate(z_test, axis=0), np.concatenate(z_gen, axis=0)
        return z_test, z_gen


class Evaluation(object):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """

    def __init__(
        self,
        generative_model,
        subset_dataset_name: str,
        gpu_device_index: int,
        config: dict,
        batch_size: int = 256,
    ):

        self.generative_model = generative_model
        self.subset_dataset_name = subset_dataset_name
        self.device = torch.device(gpu_device_index)
        self.batch_size = batch_size
        self.config = config

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(subset_dataset_name).to(self.device)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        dataset_importer = UCRDatasetImporter(subset_dataset_name, data_scaling=True)
        self.X_test = dataset_importer.X_test[:, None, :]
        n_fft = self.config["VQVAE"]["n_fft"]
        self.X_test = timefreq_to_time(
            time_to_timefreq(torch.from_numpy(self.X_test), n_fft, 1), n_fft, 1
        )
        self.X_test = self.X_test.numpy()

    def sample(
        self,
        n_samples: int,
        input_length: int,
        n_classes: int,
        kind: str,
        class_index: int = -1,
    ):
        assert kind in ["unconditional", "conditional"]

        # inference mode
        self.generative_model.eval()

        # sampling
        if kind == "unconditional":
            x_new = unconditional_sample(
                self.generative_model,
                n_samples,
                self.device,
                batch_size=self.batch_size,
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new = conditional_sample(
                self.generative_model,
                n_samples,
                self.device,
                class_index,
                self.batch_size,
            )  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        return x_new

    def compute_z(self, X_gen: torch.Tensor) -> (np.ndarray, np.ndarray):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test` and `X_gen`
        z_test, z_gen = [], []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            z_t = (
                self.fcn(
                    torch.from_numpy(self.X_test[s]).float().to(self.device),
                    return_feature_vector=True,
                )
                .cpu()
                .detach()
                .numpy()
            )
            z_g = (
                self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True)
                .cpu()
                .detach()
                .numpy()
            )

            z_test.append(z_t)
            z_gen.append(z_g)
        z_test, z_gen = np.concatenate(z_test, axis=0), np.concatenate(z_gen, axis=0)
        return z_test, z_gen

    def fid_score(
        self, z_test: np.ndarray, z_gen: np.ndarray
    ) -> (int, (np.ndarray, np.ndarray)):
        fid = calculate_fid(z_test, z_gen)
        return fid, (z_test, z_gen)

    def inception_score(self, X_gen: torch.Tensor):
        # assert self.X_test.shape[0] == X_gen.shape[0], "shape of `X_test` must be the same as that of `X_gen`."

        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `X_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(X_gen[s].float().to(self.device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen)
        return IS_mean, IS_std

    def visual_inspection(
        self, n_plot_samples: int, X_gen, ylim: tuple = (-5, 5), log=True
    ):
        # `X_test`
        sample_ind = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        for i in sample_ind:
            axes[0].plot(self.X_test[i, 0, :], alpha=0.1)
        axes[0].set_ylim(*ylim)
        plt.grid()

        # `X_gen`
        sample_ind = np.random.randint(0, X_gen.shape[0], n_plot_samples)
        for i in sample_ind:
            axes[1].plot(X_gen[i, 0, :], alpha=0.1)
        axes[1].set_ylim(*ylim)
        plt.grid()

        plt.tight_layout()
        if log:
            wandb.log({"visual inspection": wandb.Image(plt)})
            plt.close()
        else:
            plt.show()

    def pca(
        self,
        n_plot_samples: int,
        X_gen,
        z_test: np.ndarray,
        z_gen: np.ndarray,
        log=True,
    ):
        X_gen = X_gen.cpu().numpy()

        sample_ind_test = np.random.choice(
            range(self.X_test.shape[0]), size=n_plot_samples, replace=False
        )
        sample_ind_gen = np.random.choice(
            range(X_gen.shape[0]), size=n_plot_samples, replace=False
        )

        # PCA: data space
        pca = PCA(n_components=2)
        X_embedded_test = pca.fit_transform(self.X_test.squeeze()[sample_ind_test])
        X_embedded_gen = pca.transform(X_gen.squeeze()[sample_ind_gen])

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the data space")
        plt.scatter(
            X_embedded_test[:, 0], X_embedded_test[:, 1], alpha=0.1, label="test"
        )
        plt.scatter(X_embedded_gen[:, 0], X_embedded_gen[:, 1], alpha=0.1, label="gen")
        plt.legend()
        plt.tight_layout()
        if log:
            wandb.log({"PCA-data_space": wandb.Image(plt)})
            plt.close()
        else:
            plt.show()
        # PCA: latent space
        pca = PCA(n_components=2)
        z_embedded_test = pca.fit_transform(z_test.squeeze()[sample_ind_test].squeeze())
        z_embedded_gen = pca.transform(z_gen[sample_ind_gen].squeeze())

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the representation space by the trained encoder");
        plt.scatter(
            z_embedded_test[:, 0], z_embedded_test[:, 1], alpha=0.3, label="test"
        )
        plt.scatter(z_embedded_gen[:, 0], z_embedded_gen[:, 1], alpha=0.3, label="gen")
        plt.legend()
        plt.tight_layout()
        if log:
            wandb.log({"PCA-latent_space": wandb.Image(plt)})
            plt.close()
        else:
            plt.show()

    def tsne(
        self,
        n_plot_samples: int,
        X_gen,
        z_test: np.ndarray,
        z_gen: np.ndarray,
        log=True,
    ):
        X_gen = X_gen.cpu().numpy()

        sample_ind_test = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        sample_ind_gen = np.random.randint(0, X_gen.shape[0], n_plot_samples)

        # TNSE: data space
        X = np.concatenate(
            (self.X_test.squeeze()[sample_ind_test], X_gen.squeeze()[sample_ind_gen]),
            axis=0,
        ).squeeze()
        labels = np.array(["C0"] * len(sample_ind_test) + ["C1"] * len(sample_ind_gen))
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(X)

        plt.figure(figsize=(4, 4))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        if log:
            wandb.log({"TNSE-data_space": wandb.Image(plt)})
            plt.close()

        # TNSE: latent space
        Z = np.concatenate(
            (z_test[sample_ind_test], z_gen[sample_ind_gen]), axis=0
        ).squeeze()
        labels = np.array(["C0"] * len(sample_ind_test) + ["C1"] * len(sample_ind_gen))
        Z_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(Z)

        plt.figure(figsize=(4, 4))
        plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        if log:
            wandb.log({"TSNE-latent_space": wandb.Image(plt)})
            plt.close()
        else:
            plt.show()
