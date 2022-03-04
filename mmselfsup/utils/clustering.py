# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

# This file is modified from
# https://github.com/facebookresearch/deepcluster/blob/master/clustering.py

import time

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np
import torch
from scipy.sparse import csr_matrix

__all__ = ['Kmeans', 'PIC']


def preprocess_features(npdata, pca):
    """Preprocess an array of features.

    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception('nan occurs')
    if pca != -1:
        print(f'\nPCA from dim {ndim} to dim {pca}')
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                f'More than 0.1% nan occurs after pca, percent: {percent}%')
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.

    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)  # noqa E741
    return I, D


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.

    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)  # noqa E741
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print(f'k-means loss evolution: {losses}')

    return [int(n[0]) for n in I], losses[-1]


class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.labels = np.array(I)
        if verbose:
            print(f'k-means time: {time.time() - end:0.0f} s')

        return loss


def make_adjacencyW(ids, distances, sigma):
    """Create adjacency matrix with a Gaussian kernel.

    Args:
        ids (numpy array): for each vertex the ids to its nnn linked vertices
            + first column of identity.
        distances (numpy array): for each data the l2 distances to its nnn
            linked vertices + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = ids.shape
    k = k - 1
    indices = np.reshape(np.delete(ids, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(distances)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(ids, distances, sigma, alpha):
    """Run PIC algorithm."""
    a = make_adjacencyW(ids, distances, sigma)
    graph = a + a.transpose()
    nim = graph.shape[0]

    W = graph

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if (i == 200 - 1):
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for dl in range(l0, l1):
            j = W.indices[dl]
            vi = W.data[dl] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC():
    """Class to perform Power Iteration Clustering on a graph of nearest
    neighbors.

    Args:
        args: for consistency with k-means init
        sigma (float): bandwidth of the Gaussian kernel (default 0.2)
        nnn (int): number of nearest neighbors (default 5)
        alpha (float): parameter in PIC (default 0.001)
        distribute_singletons (bool): If True, reassign each singleton to
            the cluster of its closest non singleton nearest neighbors (up to
            nnn nearest neighbors).
    Attributes:
        images_lists (list of list): for each cluster, the list of image
            indexes belonging to this cluster
    """

    def __init__(self,
                 args=None,
                 sigma=0.2,
                 nnn=5,
                 alpha=0.001,
                 distribute_singletons=True,
                 pca_dim=256):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons
        self.pca_dim = pca_dim

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data, self.pca_dim)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print(f'pic time: {time.time() - end:0.0f} s')
        return 0
