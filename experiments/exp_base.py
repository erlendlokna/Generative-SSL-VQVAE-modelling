import pytorch_lightning as pl
import torch

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class ExpBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def validation_step(self, batch, batch_idx):
        raise NotImplemented

    def configure_optimizers(self):
        raise NotImplemented


def detach_the_unnecessary(loss_hist: dict):
    """
    apply `.detach()` on Tensors that do not need back-prop computation.
    :return:
    """
    for k in loss_hist.keys():
        if k not in ["loss"]:
            try:
                loss_hist[k] = loss_hist[k].detach()
            except AttributeError:
                pass


def pca_plots(zqs, y):
    pca = PCA(n_components=2)
    embs = pca.fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("PCA plot")
    plt.show()


def umap_plots(zqs, y):
    embs = umap.UMAP(densmap=True).fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("UMAP plot")
    plt.show()


def tsne_plot(zqs, y):
    embs = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("TSNE plot")
    plt.show()
