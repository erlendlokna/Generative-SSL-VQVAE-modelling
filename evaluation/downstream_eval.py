import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import umap
import umap.plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils import time_to_timefreq, quantize
import torch.nn.functional as F
import torch
import wandb
import pandas as pd
import seaborn as sns


class DownstreamEval:
    def __init__(self, train_data_loader, test_data_loader, n_fft, num_tokens):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.num_tokens = num_tokens
        self.n_fft = n_fft

    def encode_data(self, encoder, vq_model=None, device="cuda"):
        Z_tr, y_tr, train_counts = encode_data(
            dataloader=self.train_data_loader,
            encoder=encoder,
            n_fft=self.n_fft,
            vq_model=vq_model,
            device=device,
            avg_pooling=True,
            num_tokens=self.num_tokens,
        )
        Z_te, y_ts, val_counts = encode_data(
            dataloader=self.test_data_loader,
            encoder=encoder,
            n_fft=self.n_fft,
            vq_model=vq_model,
            device=device,
            avg_pooling=True,
            num_tokens=self.num_tokens,
        )

        return (
            Z_tr.cpu(),
            Z_te.cpu(),
            y_tr.cpu(),
            y_ts.cpu(),
            train_counts.cpu(),
            val_counts.cpu(),
        )

    def log_probes(self, z_tr, z_ts, y_tr, y_ts):
        scores = probes(z_tr, z_ts, y_tr, y_ts)
        wandb.log(scores)

    def log_tsne(self, z_tr, z_te, y_tr, y_te, epoch):
        z_tr = z_tr.squeeze().numpy()
        z_te = z_te.squeeze().numpy()
        y_tr = y_tr.squeeze().numpy()
        y_te = y_te.squeeze().numpy()

        tsne = TSNE(n_components=2, random_state=0)
        Z = np.concatenate((z_tr, z_te), axis=0)  # concatenate along the first axis
        Y = np.concatenate((y_tr, y_te), axis=0)  # concatenate along the first axis

        Z_tsne = tsne.fit_transform(Z)  # fit t-SNE on the concatenated array

        df = pd.DataFrame(Z_tsne, columns=["tsne-2d-one", "tsne-2d-two"])

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue=Y,
            palette=sns.color_palette("hls", len(np.unique(y_tr))),
            data=df,
            alpha=0.5,
        )
        ax.set_title("TSNE plot @ epoch: {}".format(epoch + 1))
        wandb.log({"tsne_plot": wandb.Image(fig)})
        plt.close(fig)

    def log_umap(self, z_tr, z_te, y_tr, y_te, epoch):
        z_tr = z_tr.squeeze().numpy()
        z_te = z_te.squeeze().numpy()
        y_tr = y_tr.squeeze().numpy()
        y_te = y_te.squeeze().numpy()

        Z = np.concatenate((z_tr, z_te), axis=0)  # concatenate along the first axis
        Y = np.concatenate((y_tr, y_te), axis=0)  # concatenate along the first axis

        mapper = umap.UMAP(random_state=42).fit(Z)
        f = umap.plot.points(mapper, labels=Y.reshape(-1), theme="fire")
        f.set_title("UMAP plot @ epoch: {}".format(epoch + 1))
        wandb.log({"umap_plot": wandb.Image(f)})
        plt.close()

    def log_token_usage(self, train_count, val_count, epoch):
        token_indices = np.arange(len(train_count))
        plt.figure(figsize=(10, 5))
        plt.bar(token_indices, train_count, color="dodgerblue")
        plt.xlabel("Token Index")
        plt.ylabel("Count")
        plt.xlim(-1, len(train_count))
        plt.grid(alpha=0.3)
        wandb.log({"token_train_recon_counts": wandb.Image(plt)})
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(token_indices, val_count, color="dodgerblue")
        plt.xlabel("Token Index")
        plt.ylabel("Count")
        plt.xlim(-1, len(train_count))
        plt.grid(alpha=0.3)
        wandb.log({"token_val_recon_counts": wandb.Image(plt)})
        plt.close()

    def log_codebook_similarity(self, codebook, epoch, device):
        corr = torch.corrcoef(codebook).to(device)  # correlation matrix
        cos_sim = F.cosine_similarity(
            codebook.unsqueeze(0), codebook.unsqueeze(1), dim=2
        ).to(
            device
        )  # cosine similarity matrix

        mean_abs_corr_off_diagonal = torch.sum(
            torch.abs(corr - torch.eye(corr.shape[0]).to(device))
        ) / (corr.shape[0] * (corr.shape[0] - 1))

        mean_abs_cos_sim_off_diagonal = torch.sum(
            torch.abs(cos_sim - torch.eye(cos_sim.shape[0]).to(device))
        ) / (cos_sim.shape[0] * (cos_sim.shape[0] - 1))

        wandb.log({"mean_abs_corr_off_diagonal": mean_abs_corr_off_diagonal})
        wandb.log({"mean_abs_cos_sim_off_diagonal": mean_abs_cos_sim_off_diagonal})

        corr_viz = corr.cpu().numpy()
        cos_sim_viz = cos_sim.cpu().numpy()
        # Set the diagonal elements of corr_viz to np.nan for visualization
        np.fill_diagonal(corr_viz, np.nan)
        np.fill_diagonal(cos_sim_viz, np.nan)

        # im = plt.imshow(corr_viz)
        sns.heatmap(corr_viz, cmap="magma")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        plt.title(
            f"Mean absolute off-diagonal correlation (@{epoch}): {np.round(mean_abs_corr_off_diagonal.cpu(), 4)}"
        )
        wandb.log({"correlation_matrix": wandb.Image(plt)})
        plt.close()

        sns.heatmap(cos_sim_viz, cmap="magma")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        plt.title(
            f"Mean absolute off-diagonal cosine similarity (@{epoch}): {np.round(mean_abs_cos_sim_off_diagonal.cpu(), 4)}"
        )
        wandb.log({"cosine_similarity_matrix": wandb.Image(plt)})
        plt.close()

        # histogram of codebook correlation
        n = corr.shape[0]
        corr = (
            corr.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        )  # remove the diagonal elements
        corr = corr.view(-1)  # flatten
        sns.histplot(corr.cpu().numpy(), bins=100)
        plt.xlabel("Correlation")
        plt.title(f"histogram of codebook correlation (@{epoch})")
        wandb.log({"corr_hist": wandb.Image(plt)})

    def log_corr_vs_usage(self, codebook, train_count, epoch):
        usage_ratio = train_count / torch.sum(train_count)

        corr = torch.abs(torch.corrcoef(codebook))

        # Remove diagonal elements
        corr = corr - torch.diag(torch.diag(corr))

        total_corr_per = torch.sum(corr, dim=0)

        most_used = torch.argsort(usage_ratio)
        corresponding_corr = total_corr_per[most_used]

        df = pd.DataFrame(
            {
                "Token Usage Ratio In Reconstruction": usage_ratio.cpu().numpy(),
                "Abs Correlation": corresponding_corr.cpu().numpy(),
            }
        )
        sns.jointplot(
            data=df,
            x="Token Usage Ratio In Reconstruction",
            y="Abs Correlation",
            label="token",
        )
        wandb.log({"corr_vs_usage": wandb.Image(plt)})
        plt.close()


def encode_data(
    dataloader,
    encoder,
    n_fft,
    vq_model=None,
    avg_pooling=False,
    num_tokens=32,
    device="cuda",
):
    """
    Function to encode the data using the encoder and optionally the quantizer.
    It encodes to continous latent variables by default (vq_model=False).
    ---
    returns
    """
    z_list = []  # List to hold all the encoded representations
    y_list = []  # List to hold all the labels/targets

    # Iterate over the entire dataloader
    counts = torch.zeros(num_tokens, device=device)

    for batch in dataloader:
        x, y = batch  # Unpack the batch.
        if len(x) == 2:
            x = x[0]  # discard the potential augmented view
        x = x.to(device)
        # Perform the encoding
        C = x.shape[1]
        xf = time_to_timefreq(x, n_fft, C).to(
            device
        )  # Convert time domain to frequency domain
        z = encoder(xf.to(device)).detach()  # Encode the input

        if vq_model is not None:
            z, s, _, _ = quantize(z, vq_model)
            counts += torch.bincount(s.flatten(), minlength=32)

        # Convert the tensors to lists and append to z_list and y_list
        z_list.extend(z.cpu().detach().tolist())
        y_list.extend(
            y.cpu().detach().tolist()
        )  # Make sure to detach y and move to CPU as well

    # Convert lists of lists to 2D tensors
    z_encoded = torch.tensor(z_list, device=device)
    ys = torch.tensor(y_list, device=device)

    if avg_pooling:
        z_encoded = F.adaptive_avg_pool2d(z_encoded, (1, 1)).squeeze(-1).squeeze(-1)

    return z_encoded, ys, counts


def probes(x_tr, x_ts, y_tr, y_ts):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_ts = scaler.transform(x_ts)
    y_tr = y_tr.flatten()
    y_ts = y_ts.flatten()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_tr, y_tr)
    preds_knn = knn.predict(x_ts)

    svm = SVC(kernel="linear")
    svm.fit(x_tr, y_tr)
    preds_svm = svm.predict(x_ts)

    scores = {
        "knn_accuracy": metrics.accuracy_score(y_ts, preds_knn),
        "svm_accuracy": metrics.accuracy_score(y_ts, preds_svm),
    }
    return scores
