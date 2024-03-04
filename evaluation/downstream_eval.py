import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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

    scores = {"knn_accuracy": metrics.accuracy_score(y_ts, preds_knn),
               "svm_accuracy": metrics.accuracy_score(y_ts, preds_svm)}

    
    return scores

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
