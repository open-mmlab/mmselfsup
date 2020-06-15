import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..registry import MEMORIES


@MEMORIES.register_module
class ODCMemoryGPU(nn.Module):
    '''Memory bank for Online Deep Clustering. Feature bank stored in GPU.
    '''

    def __init__(self, length, feat_dim, momentum, num_classes, min_cluster,
                 **kwargs):
        super(ODCMemoryGPU, self).__init__()
        self.rank, self.num_replicas = get_dist_info()
        self.feature_bank = torch.zeros((length, feat_dim),
                                        dtype=torch.float32).cuda()
        self.label_bank = torch.zeros((length, ), dtype=torch.long).cuda()
        self.centroids = torch.zeros((num_classes, feat_dim),
                                     dtype=torch.float32).cuda()
        self.kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20)
        self.feat_dim = feat_dim
        self.initialized = False
        self.momentum = momentum
        self.num_classes = num_classes
        self.min_cluster = min_cluster
        self.debug = kwargs.get('debug', False)

    @torch.no_grad()
    def init_memory(self, feature, label):
        self.initialized = True
        self.label_bank.copy_(torch.from_numpy(label).long().cuda())
        # make sure no empty clusters
        assert (np.bincount(label, minlength=self.num_classes) != 0).all()
        feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10)
        self.feature_bank.copy_(torch.from_numpy(feature))
        self._compute_centroids()

    @torch.no_grad()
    def _compute_centroids_ind(self, cinds):
        '''compute a few centroids'''
        for i, c in enumerate(cinds):
            ind = torch.where(self.label_bank == c)[0]
            self.centroids[i, :] = self.feature_bank[ind, :].mean(dim=0)

    def _compute_centroids(self):
        if self.debug:
            print("enter: _compute_centroids")
        '''compute all non-empty centroids'''
        l = self.label_bank.cpu().numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        for i, st, ed in zip(class_start, start, end):
            self.centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(
                dim=0)

    def _gather(self, ind, feature):  # gather ind and feature
        if self.debug:
            print("enter: _gather")
        assert ind.size(0) > 0
        ind_gathered = [
            torch.ones_like(ind).cuda() for _ in range(self.num_replicas)
        ]
        feature_gathered = [
            torch.ones_like(feature).cuda() for _ in range(self.num_replicas)
        ]
        dist.all_gather(ind_gathered, ind)
        dist.all_gather(feature_gathered, feature)
        ind_gathered = torch.cat(ind_gathered, dim=0)
        feature_gathered = torch.cat(feature_gathered, dim=0)
        return ind_gathered, feature_gathered

    def update_samples_memory(self, ind, feature):  # ind, feature: cuda tensor
        if self.debug:
            print("enter: update_samples_memory")
        assert self.initialized
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        ind, feature_norm = self._gather(
            ind, feature_norm)  # ind: (N*w), feature: (N*w)xk, cuda tensor
        # momentum update
        feature_old = self.feature_bank[ind, ...]
        feature_new = (1 - self.momentum) * feature_old + \
            self.momentum * feature_norm
        feature_norm = feature_new / (
            feature_new.norm(dim=1).view(-1, 1) + 1e-10)
        self.feature_bank[ind, ...] = feature_norm
        # compute new labels
        similarity_to_centroids = torch.mm(self.centroids,
                                           feature_norm.permute(1, 0))  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        change_ratio = (newlabel !=
            self.label_bank[ind]).sum().float() \
            / float(newlabel.shape[0])
        self.label_bank[ind] = newlabel.clone()  # copy to cpu
        return change_ratio

    @torch.no_grad()
    def deal_with_small_clusters(self):
        if self.debug:
            print("enter: deal_with_small_clusters")
        # check empty class
        hist = torch.bincount(self.label_bank, minlength=self.num_classes)
        small_clusters = torch.where(hist < self.min_cluster)[0]
        if self.debug and self.rank == 0:
            print("mincluster: {}, num of small class: {}".format(
                hist.min(), len(small_clusters)))
        if len(small_clusters) == 0:
            return
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = torch.where(self.label_bank == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_classes),
                        small_clusters.cpu().numpy(),
                        assume_unique=True)).cuda()
                target_ind = torch.mm(self.centroids[inclusion, :],
                                      self.feature_bank[ind, :].permute(
                                          1, 0)).argmax(dim=0)
                target = inclusion[target_ind]
                self.label_bank[ind] = target
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)

    def update_centroids_memory(self, cinds=None):
        if cinds is None:
            self._compute_centroids()
        else:
            self._compute_centroids_ind(cinds)

    def _partition_max_cluster(self, max_cluster):
        if self.debug:
            print("enter: _partition_max_cluster")
        assert self.rank == 0  # avoid randomness among ranks
        max_cluster_inds = torch.where(self.label_bank == max_cluster)[0]
        size = len(max_cluster_inds)

        assert size >= 2  # image indices in the max cluster
        max_cluster_features = self.feature_bank[max_cluster_inds, :]
        if torch.any(torch.isnan(max_cluster_features)):
            raise Exception("Has nan in features.")
        kmeans_ret = self.kmeans.fit(max_cluster_features.cpu().numpy())
        kmeans_labels = torch.from_numpy(kmeans_ret.labels_).cuda()
        sub_cluster1_ind = max_cluster_inds[kmeans_labels == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_labels == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                "Warning: kmeans partition fails, resort to random partition.")
            rnd_idx = torch.randperm(size)
            sub_cluster1_ind = max_cluster_inds[rnd_idx[:size // 2]]
            sub_cluster2_ind = max_cluster_inds[rnd_idx[size // 2:]]
        return sub_cluster1_ind, sub_cluster2_ind

    def _redirect_empty_clusters(self, empty_clusters):
        if self.debug:
            print("enter: _redirect_empty_clusters")
        for e in empty_clusters:
            assert (self.label_bank != e).all().item(), \
                "Cluster #{} is not an empty cluster.".format(e)
            max_cluster = torch.bincount(
                self.label_bank, minlength=self.num_classes).argmax().item()
            # gather partitioning indices
            if self.rank == 0:
                sub_cluster1_ind, sub_cluster2_ind = self._partition_max_cluster(
                    max_cluster)
                size2 = torch.LongTensor([len(sub_cluster2_ind)]).cuda()
            else:
                size2 = torch.LongTensor([0]).cuda()
            dist.all_reduce(size2)
            if self.rank != 0:
                sub_cluster2_ind = torch.zeros((size2, ),
                                               dtype=torch.int64).cuda()
            dist.broadcast(sub_cluster2_ind, 0)

            # reassign samples in partition #2 to the empty class
            self.label_bank[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_centroids_memory([max_cluster, e])
