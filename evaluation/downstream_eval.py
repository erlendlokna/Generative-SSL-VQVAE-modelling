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

from models.stage2.maskgit import MaskGIT
from preprocessing.preprocess_ucr import UCRDatasetImporter
from models.stage2.sample import unconditional_sample, conditional_sample
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


def probes(summary_tr, summary_ts, y_tr, y_ts):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(summary_tr)
    summary_tr = scaler.transform(summary_tr)
    summary_ts = scaler.transform(summary_ts)
    y_tr = y_tr.flatten()
    y_ts = y_ts.flatten()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(summary_tr, y_tr)
    preds = knn.predict(summary_ts)

    scores = {"knn_accuracy": metrics.accuracy_score(y_ts, preds)}

    return scores
