# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

# This file is modified from
# https://github.com/facebookresearch/deepcluster/blob/master/clustering.py
import time
from typing import Any, List, Optional, Tuple

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np
import torch
from scipy.sparse import csr_matrix

__all__ = ['Kmeans', 'PIC']


def preprocess_features(npdata, pca: np.ndarray) -> np.ndarray:
    """Preprocess an array of features.

    Args:
        npdata (np.ndarray): Features to preprocess.
        pca (int): Dim of output.

    Returns:
        np.ndarray: Data PCA-reduced, whitened and L2-normalized, with dim
            N * pca.
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


def make_graph(xb: np.ndarray, nnn: int) -> Tuple[List, List]:
    """Builds a graph of nearest neighbors.

    Args:
        xb (np.ndarray): Input data.
        nnn (int): Number of nearest neighbors.

    Returns:
        Tuple[List, List]:
            - I: for each data the list of distances to its nnn NN.
            - D: for each data the list of ids to its nnn nearest neighbors.
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


def run_kmeans(x: np.ndarray,
               nmb_clusters: int,
               verbose: Optional[bool] = False) -> Tuple[List, float]:
    """Runs kmeans on 1 GPU.

    Args:
        x (np.ndarray): Data.
        nmb_clusters (int): Number of clusters.
        verbose (bool, optional): Whether to print information.

    Returns:
        Tuple[List, float]:
            - List: ids of data in each cluster.
            - losses: loss of clustering.
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

    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    if verbose:
        print(f'k-means loss evolution: {losses}')

    return [int(n[0]) for n in I], losses[-1]


class Kmeans:
    """K-means algorithm process.

    Args:
        k (int): Number of clusters.
        pca_dim: Dim of output.
    """

    def __init__(self, k: int, pca_dim: Optional[int] = 256) -> None:
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat: np.ndarray, verbose=False) -> float:
        """Performs k-means clustering.

        Args:
            feat (np.ndarray): data to cluster, with N * dim.
            verbose (bool, optional): Whether to print information.

        Returns:
            float: Loss of clustering.
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


def make_adjacencyW(ids: np.ndarray, distances: np.ndarray,
                    sigma: float) -> csr_matrix:
    """Create adjacency matrix with a Gaussian kernel.

    Args:
        ids (np.ndarray): For each vertex the ids to its nnn linked vertices
            + first column of identity.
        distances (np.ndarray): For each data the l2 distances to its nnn
            linked vertices + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: Affinity matrix of the graph.
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


def run_pic(ids: np.ndarray, distances: np.ndarray, sigma: float,
            alpha: float) -> List[int]:
    """Run PIC algorithm.

    Args:
        ids (np.ndarray): For each vertex the ids to its nnn linked vertices
            + first column of identity.
        distances (np.ndarray): For each data the l2 distances to its nnn
            linked vertices + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.
        alpha (float): Parameter in PIC.

    Returns:
        List[int]: Cluster information.
    """
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


def find_maxima_cluster(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Find maxima cluster in PIC algorithm."""
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
        args (optional): For consistency with k-means init
        sigma (float, optional): Bandwidth of the Gaussian kernel.
            Defaults to 0.2.
        nnn (int, optional): Number of nearest neighbors. Defaults to 5.
        alpha (float, optional): Parameter in PIC. Defaults to 0.001.
        distribute_singletons (bool, optional): If True, reassign each
            singleton to the cluster of its closest non singleton nearest
            neighbors (up to nnn nearest neighbors). Defaults to True.
        pca_dim (int, optional): Dim of output. Defaults to 256.
    """

    def __init__(self,
                 args: Any = None,
                 sigma: Optional[float] = 0.2,
                 nnn: Optional[int] = 5,
                 alpha: Optional[float] = 0.001,
                 distribute_singletons: Optional[bool] = True,
                 pca_dim: Optional[int] = 256) -> None:
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons
        self.pca_dim = pca_dim

    def cluster(self, data, verbose=False):
        """Performs PIC clustering.

        Args:
            data (np.ndarray): data to cluster, with N * dim.
            verbose (bool, optional): Whether to print information.

        Returns:
            int: 0.
        """
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
