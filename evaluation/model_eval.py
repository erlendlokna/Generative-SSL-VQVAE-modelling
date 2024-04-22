"""
FID, IS, JS divergence.
"""

import numpy as np
import os
import copy
import tempfile
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import kurtosis, skew
from scipy.special import entr

from models.stage2.maskgit import MaskGIT
from models.stage2.full_embedding_maskgit import Full_Embedding_MaskGIT
from preprocessing.preprocess_ucr import UCRDatasetImporter
from models.stage2.sample import unconditional_sample, conditional_sample
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score
from utils import time_to_timefreq, timefreq_to_time, model_filename
import seaborn as sns
import pandas as pd


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
        subset_dataset_name: str,
        gpu_device_index: int,
        config: dict,
        batch_size: int = 256,
    ):
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

    def sampleMaskGit(
        self,
        n_samples: int,
        input_length: int,
        n_classes: int,
        kind: str,
        class_index: int = -1,
    ):
        assert kind in ["unconditional", "conditional"]

        # build
        maskgit = MaskGIT(
            input_length,
            **self.config["MaskGIT"],
            config=self.config,
            n_classes=n_classes,
        ).to(self.device)

        # load
        fname = (
            f"{model_filename(self.config, 'maskgit')}-{self.subset_dataset_name}.ckpt"
        )
        try:
            ckpt_fname = os.path.join("saved_models", fname)
            maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
        except FileNotFoundError:
            ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
            maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)

        # inference mode
        maskgit.eval()

        # sampling
        if kind == "unconditional":
            x_new = unconditional_sample(
                maskgit, n_samples, self.device, batch_size=self.batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new = conditional_sample(
                maskgit, n_samples, self.device, class_index, self.batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        return x_new

    def sampleFullEmbedMaskGit(
        self,
        n_samples: int,
        input_length: int,
        n_classes: int,
        kind: str,
        device: torch.device,
        class_index: int = -1,
        load_finetuned_codebook: bool = False,
    ):
        assert kind in ["unconditional", "conditional"]

        # build
        maskgit = Full_Embedding_MaskGIT(
            input_length,
            **self.config["MaskGIT"],
            config=self.config,
            n_classes=n_classes,
            finetune_codebook=False,
            device=device,
            load_finetuned_codebook=load_finetuned_codebook,
        ).to(self.device)

        # load
        model_name = "fullembed-maskgit"
        if load_finetuned_codebook:
            model_name += "-finetuned"

        fname = (
            f"{model_filename(self.config, model_name)}-{self.subset_dataset_name}.ckpt"
        )
        try:
            ckpt_fname = os.path.join("saved_models", fname)
            maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
        except FileNotFoundError:
            ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
            maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)

        # inference mode
        maskgit.eval()

        # sampling
        if kind == "unconditional":
            x_new = unconditional_sample(
                maskgit, n_samples, self.device, batch_size=self.batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new = conditional_sample(
                maskgit, n_samples, self.device, class_index, self.batch_size
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

    def log_visual_inspection(self, n_plot_samples: int, X_gen, ylim: tuple = (-5, 5)):
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
        wandb.log({"visual inspection": wandb.Image(plt)})
        plt.close()

    def log_pca(
        self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray
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
        wandb.log({"PCA-data_space": wandb.Image(plt)})
        plt.close()

        # PCA: latent space
        pca = PCA(n_components=2)
        z_embedded_test = pca.fit_transform(z_test.squeeze()[sample_ind_test].squeeze())
        z_embedded_gen = pca.transform(z_gen[sample_ind_gen].squeeze())

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the representation space by the trained encoder");
        plt.scatter(
            z_embedded_test[:, 0], z_embedded_test[:, 1], alpha=0.1, label="test"
        )
        plt.scatter(z_embedded_gen[:, 0], z_embedded_gen[:, 1], alpha=0.1, label="gen")
        plt.legend()
        plt.tight_layout()
        wandb.log({"PCA-latent_space": wandb.Image(plt)})
        plt.close()

    def log_tsne(
        self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray
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
        wandb.log({"TSNE-latent_space": wandb.Image(plt)})
        plt.close()

    def aggregate_statistics(self, model, num_iterations=100):
        results = []

        for _ in tqdm(range(num_iterations)):
            sample, probs_array, entropy_array, sel_entropy_array = (
                model.iterative_decoding(stats=True)
            )
            results.append(
                (sample[-1].squeeze(), probs_array, entropy_array, sel_entropy_array)
            )

        # Aggregate results across iterations-
        aggregated_probs = torch.stack(
            [torch.stack(probs) for _, probs, _, _ in results]
        )
        aggregated_entropy = torch.stack(
            [torch.stack(entropy) for _, _, entropy, _ in results]
        )
        aggregated_sel_entropy = torch.stack(
            [torch.stack(sel_entropy) for _, _, _, sel_entropy in results]
        )

        aggregated_samples = torch.stack([sample for sample, _, _, _ in results])

        # broadcasting: (num_iterations, t, 1, num tokens) -> (num_iterations, t, num_tokens)
        aggregated_probs = torch.squeeze(aggregated_probs, dim=2)
        aggregated_entropy = torch.squeeze(aggregated_entropy, dim=2)
        # aggregated_samples = torch.squeeze(aggregated_samples, dim=2)
        # aggregated_sel_entropy = torch.squeeze(aggregated_sel_entropy, dim=2)
        return (
            aggregated_samples,
            aggregated_probs,
            aggregated_entropy,
            aggregated_sel_entropy,
        )

    def log_conditional_probability_vs_t(self, agg_probs):
        a, b, c = agg_probs.shape
        flat_agg_probs = agg_probs.permute(1, 0, 2).reshape(b, a * c)

        # Calculate the means and standard deviations along the second axis
        means = []
        stds = []

        for i in range(flat_agg_probs.shape[0]):
            probs = flat_agg_probs[i]

            means.append(torch.mean(probs[torch.isfinite(probs)]))
            stds.append(torch.std(probs[torch.isfinite(probs)]))

        # Create an array for the "t" values
        t_values = np.arange(1, len(means) + 1)

        # Plot the means and standard deviations
        f, ax = plt.subplots()
        ax.errorbar(
            t_values,
            means,
            yerr=stds,
            fmt="o",
            color="dodgerblue",
            alpha=0.8,
            label="mean conditional probability",
        )

        means = np.stack(means)
        stds = np.stack(stds)

        ax.set_title("conditional probability")
        ax.set_xlabel("t")
        ax.set_ylabel("p")
        ax.grid(alpha=0.3)
        wandb.log({"uncond_p_vs_t": wandb.Image(f)})
        plt.close()

        return means, stds

    def log_entropy_vs_t(self, agg_entropy):
        a, b, c = agg_entropy.shape
        flat_agg_entropy = agg_entropy.permute(1, 0, 2).reshape(b, a * c)

        # Calculate the means and standard deviations along the second axis
        means = []
        stds = []

        for i in range(flat_agg_entropy.shape[0]):
            entropies = flat_agg_entropy[i]

            means.append(torch.mean(entropies))
            stds.append(torch.std(entropies))

        # Create an array for the "t" values
        t_values = np.arange(1, len(means) + 1)

        # Plot the means and standard deviations
        f, ax = plt.subplots()
        ax.errorbar(
            t_values,
            means,
            yerr=stds,
            fmt="o",
            color="dodgerblue",
            alpha=0.8,
            label="mean entropy",
        )

        ax.legend()
        ax.set_title("Mean and Std of entropies vs t")
        ax.set_xlabel("t")
        ax.set_ylabel("entropy")
        ax.grid(alpha=0.3)
        wandb.log({"entropy_vs_t": wandb.Image(f)})
        plt.close()

        means = np.stack(means)
        stds = np.stack(stds)

        # Return the figure and axes
        return means, stds

    def log_selected_entropy_vs_t(self, agg_sel_entropy):
        means = torch.mean(agg_sel_entropy.T, dim=1).cpu().numpy()
        stds = torch.std(agg_sel_entropy.T, dim=1).cpu().numpy()

        t_values = np.arange(1, len(means) + 1)
        f, ax = plt.subplots()
        ax.errorbar(
            t_values,
            means,
            yerr=stds,
            fmt="o",
            color="dodgerblue",
            alpha=0.8,
            label="mean entropy",
        )
        ax.set_title("Entropy of selected tokens vs t")
        ax.set_xlabel("t")
        ax.set_ylabel("entropy")
        ax.grid(alpha=0.3)
        wandb.log({"sel_entropy_vs_t": wandb.Image(f)})
        plt.close()
        return means, stds

    def log_probability_histograms(self, agg_probs):
        num_tokens = agg_probs.shape[1]

        f, axs = plt.subplots(num_tokens, 1, figsize=(5, 15))
        for i in range(num_tokens):
            # Filter out infinite and NaN values
            probs = agg_probs[:, i].flatten().cpu().numpy()
            finite_probs = probs[np.isfinite(probs) & ~np.isnan(probs)]

            axs[i].hist(finite_probs, bins=100, alpha=0.5)
            axs[i].set_title(f"t =  {i+1}")
            axs[i].set_xlabel("Selected Probabilities")
            axs[i].set_ylabel("Frequency")
            axs[i].set_xlim([0, 1])
            # axs[i].set_ylim([0, 400])

        plt.tight_layout()
        wandb.log({"probability_histograms": wandb.Image(f)})
        plt.close()

    def log_iterative_decoding_statistics(
        self, agg_sel_probs, agg_entropy, agg_sel_entropy
    ):
        self.log_probability_histograms(agg_sel_probs)

        mean_p_t, std_p_t = self.log_conditional_probability_vs_t(agg_sel_probs)

        mean_e_t, std_e_t = self.log_entropy_vs_t(agg_entropy)

        mean_sel_e_t, std_sel_e_t = self.log_selected_entropy_vs_t(agg_sel_entropy)

        data = np.array(
            [
                mean_p_t,
                std_p_t,
                mean_e_t,
                std_e_t,
                mean_sel_e_t,
                std_sel_e_t,
            ]
        ).T.tolist()

        iterative_decoding = wandb.Table(
            data=data,
            columns=[
                "p_cond_mean",
                "p_cond_std",
                "e_mean",
                "e_std",
                "sel_e_mean",
                "sel_e_std",
            ],
        )

        wandb.log({"iterative_decoding": iterative_decoding})

    def gini_coefficient(self, array):
        array = array.flatten()
        if torch.any(array < 0):
            array = array[array > 0]
        array += 0.0000001  # Avoid division by zero
        array = torch.sort(array)[0]
        index = torch.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (
            (torch.sum((2 * index - n - 1) * array)) / (n * torch.sum(array))
        ).item()

    def log_coverage_and_variety(self, aggregated_samples, vocabulary_size=32):
        num_samples, sequence_length = aggregated_samples.size()

        # Coverage Score
        coverage_scores = torch.tensor(
            [
                len(torch.unique(aggregated_samples[i])) / vocabulary_size
                for i in range(num_samples)
            ]
        )
        coverage_score = torch.mean(coverage_scores)
        coverage_std = torch.std(coverage_scores)

        # Variety Score via Gini Coefficient
        token_counts = torch.bincount(aggregated_samples.view(-1))
        variety_score = self.gini_coefficient(token_counts.float())

        # Entropy
        probabilities = token_counts.float() / token_counts.sum()
        entropy = torch.sum(entr(probabilities))

        # Kurtosis and Skewness
        kurt = kurtosis(token_counts.float())
        skewness = skew(token_counts.float())

        wandb.log(
            {
                "coverage": coverage_score.item(),
                "coverage_std": coverage_std.item(),
                "variety (Gini)": variety_score,
                "entropy": entropy.item(),
                "kurtosis": kurt,
                "skewness": skewness,
            }
        )

    def log_prior_token_ratios(self, token_probs):

        token_indices = np.arange(len(token_probs))  # Token indices for the x-axis

        plt.figure(figsize=(10, 5))
        sns.barplot(x=token_indices, y=token_probs.numpy(), color="dodgerblue")

        plt.title("Token Sampling Ratio")
        plt.xlabel("Token Index")
        plt.ylabel("Ratio")
        plt.xlim(-1, len(token_probs))
        plt.grid(alpha=0.3)
        wandb.log({"prior_token_ratios": wandb.Image(plt)})
        plt.close()

        return token_probs

    def co_occurence_matrix(self, samples, num_tokens):

        co_occurrence = torch.zeros((num_tokens, num_tokens))

        for sequence in samples:
            sequence = sequence.float()
            # Count the occurrence of each token in the sequence
            token_counts = torch.histc(
                sequence, bins=num_tokens, min=0, max=num_tokens - 1
            )
            # Update the co-occurrence matrix
            co_occurrence += torch.outer(token_counts, token_counts)

        # Divide by 2 to correct for double counting
        co_occurrence /= 2

        return co_occurrence

    def log_co_occurence(self, co_occurence):
        sns.heatmap(co_occurence, cmap="magma")
        plt.title("Co-Occurrence Matrix of Tokens")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        wandb.log({"co_occurence": wandb.Image(plt)})
        plt.close()

    def calculate_probabilities(self, co_occurence):

        total_sum = torch.sum(co_occurence)

        token_prob = torch.sum(co_occurence, dim=0) / total_sum

        joint_prob = co_occurence / total_sum

        conditional_prob = joint_prob / token_prob[None, :]

        return token_prob, joint_prob, conditional_prob

    def log_conditional_probs(self, conditional_prob):
        sns.heatmap(conditional_prob)
        plt.title("Conditional Probability of Tokens")
        wandb.log({"conditional_prob": wandb.Image(plt)})
        plt.close()

    def calculate_pmi(self, token_prob, joint_prob):
        # Following formula: PMI = log2(P(x,y) / (P(x) * P(y)))

        eps = 1e-10

        token_prob = token_prob / token_prob.sum()
        joint_prob = joint_prob / joint_prob.sum()

        # Make sure the denominator is non-zero
        denominator = torch.outer(token_prob, token_prob)  # P(x) * P(y)
        denominator = torch.clamp(denominator, min=eps)  # Ensure it's non-zero.

        mutual_info = joint_prob / denominator  # P(x,y) / (P(x) * P(y))
        mutual_info = torch.clamp(
            mutual_info, min=eps
        )  # Ensure it's non-zero before taking log
        mutual_info = torch.log2(mutual_info)  # log2(P(x,y) / (P(x) * P(y)))

        mutual_info = torch.clamp(
            mutual_info, min=0.0
        )  # Not interested in negative values

        return mutual_info

    def log_pmi(self, pmi):
        # Plot the PMI matrix without diag
        pmi_no_diag = pmi.fill_diagonal_(0)
        sns.heatmap(pmi, cmap="coolwarm")
        plt.title("Off Diagonal Pointwise Mutual Information of Tokens")
        wandb.log({"pmi": wandb.Image(plt)})
        plt.close()

        pmi_no_diag = pmi_no_diag.flatten()
        pmi_no_diag = pmi_no_diag[pmi_no_diag > 0]
        sns.histplot(pmi_no_diag.cpu().numpy(), bins=100)
        plt.xlabel("PMI > 0")
        plt.title("Histogram of PMI values")
        wandb.log({"pmi_hist": wandb.Image(plt)})
        plt.close()

    def log_pmi_vs_usage(self, pmi, token_prob):
        # Calculating total pointwise mutual information (TPMI) for each token
        tpmi = pmi.sum(axis=0)

        # Sorting indices based on token probability for ascending order
        sorted_indices = torch.argsort(token_prob)

        # Sorting the token probabilities and corresponding TPMI values
        most_sampled = token_prob[sorted_indices].numpy()
        corr_tpmi = tpmi[sorted_indices].numpy()

        # Creating a DataFrame for easier plotting
        df = pd.DataFrame({"Sample Usage Ratio": most_sampled, "TPMI": corr_tpmi})

        # Creating the plot with seaborn
        sns.jointplot(
            data=df,
            x="Sample Usage Ratio",
            y="TPMI",
            color="royalblue",
            height=6,
        )

        # Enhancing the plot
        plt.title("Sample Usage Ratio vs TPMI", pad=0)
        plt.xlabel("Sample Usage Ratio")
        plt.ylabel("TPMI")

        # Optionally, annotate some points
        # Example: Highlight the token with the highest TPMI
        min_tpmi_idx = df["TPMI"].idxmin()
        plt.scatter(
            df.iloc[min_tpmi_idx]["Sample Usage Ratio"],
            df.iloc[min_tpmi_idx]["TPMI"],
            color="red",
        )
        plt.text(
            df.iloc[min_tpmi_idx]["Sample Usage Ratio"],
            df.iloc[min_tpmi_idx]["TPMI"],
            "Lowest TPMI",
            color="red",
            verticalalignment="bottom",
        )

        # Show the plot
        if max(df["Sample Usage Ratio"]) < 0.4:
            plt.xlim(0, 0.4)

        plt.tight_layout()
        wandb.log({"pmi_vs_usage": wandb.Image(plt)})
        plt.close()
