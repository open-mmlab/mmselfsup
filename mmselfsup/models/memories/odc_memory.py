# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, get_dist_info
from sklearn.cluster import KMeans

from ..builder import MEMORIES


@MEMORIES.register_module()
class ODCMemory(BaseModule):
    """Memory module for ODC.

    This module includes the samples memory and the centroids memory in ODC.
    The samples memory stores features and pseudo-labels of all samples in the
    dataset; while the centroids memory stores features of cluster centroids.

    Args:
        length (int): Number of features stored in samples memory.
        feat_dim (int): Dimension of stored features.
        momentum (float): Momentum coefficient for updating features.
        num_classes (int): Number of clusters.
        min_cluster (int): Minimal cluster size.
    """

    def __init__(self, length, feat_dim, momentum, num_classes, min_cluster,
                 **kwargs):
        super(ODCMemory, self).__init__()
        self.rank, self.num_replicas = get_dist_info()
        if self.rank == 0:
            self.feature_bank = torch.zeros((length, feat_dim),
                                            dtype=torch.float32)
        self.label_bank = torch.zeros((length, ), dtype=torch.long)
        self.centroids = torch.zeros((num_classes, feat_dim),
                                     dtype=torch.float32).cuda()
        self.kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20)
        self.feat_dim = feat_dim
        self.initialized = False
        self.momentum = momentum
        self.num_classes = num_classes
        self.min_cluster = min_cluster
        self.debug = kwargs.get('debug', False)

    def init_memory(self, feature, label):
        """Initialize memory modules."""
        self.initialized = True
        self.label_bank.copy_(torch.from_numpy(label).long())
        # make sure no empty clusters
        assert (np.bincount(label, minlength=self.num_classes) != 0).all()
        if self.rank == 0:
            feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10)
            self.feature_bank.copy_(torch.from_numpy(feature))
            centroids = self._compute_centroids()
            self.centroids.copy_(centroids)
        dist.broadcast(self.centroids, 0)

    def _compute_centroids_ind(self, cinds):
        """Compute a few centroids."""
        assert self.rank == 0
        num = len(cinds)
        centroids = torch.zeros((num, self.feat_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            ind = np.where(self.label_bank.numpy() == c)[0]
            centroids[i, :] = self.feature_bank[ind, :].mean(dim=0)
        return centroids

    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        assert self.rank == 0
        label_bank_np = self.label_bank.numpy()
        argl = np.argsort(label_bank_np)
        sortl = label_bank_np[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(label_bank_np))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return centroids

    def _gather(self, ind, feature):
        """Gather indices and features."""
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

    def update_samples_memory(self, ind, feature):
        """Update samples memory."""
        assert self.initialized
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        ind, feature_norm = self._gather(
            ind, feature_norm)  # ind: (N*w), feature: (N*w)xk, cuda tensor
        ind = ind.cpu()
        if self.rank == 0:
            feature_old = self.feature_bank[ind, ...].cuda()
            feature_new = (1 - self.momentum) * feature_old + \
                self.momentum * feature_norm
            feature_norm = feature_new / (
                feature_new.norm(dim=1).view(-1, 1) + 1e-10)
            self.feature_bank[ind, ...] = feature_norm.cpu()
        dist.barrier()
        dist.broadcast(feature_norm, 0)
        # compute new labels
        similarity_to_centroids = torch.mm(self.centroids,
                                           feature_norm.permute(1, 0))  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        newlabel_cpu = newlabel.cpu()
        change_ratio = (newlabel_cpu != self.label_bank[ind]
                        ).sum().float().cuda() / float(newlabel_cpu.shape[0])
        self.label_bank[ind] = newlabel_cpu.clone()  # copy to cpu
        return change_ratio

    def deal_with_small_clusters(self):
        """Deal with small clusters."""
        # check empty class
        histogram = np.bincount(
            self.label_bank.numpy(), minlength=self.num_classes)
        small_clusters = np.where(histogram < self.min_cluster)[0].tolist()
        if self.debug and self.rank == 0:
            print(f'mincluster: {histogram.min()}, '
                  f'num of small class: {len(small_clusters)}')
        if len(small_clusters) == 0:
            return
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = np.where(self.label_bank.numpy() == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_classes),
                        np.array(small_clusters),
                        assume_unique=True)).cuda()
                if self.rank == 0:
                    target_ind = torch.mm(
                        self.centroids[inclusion, :],
                        self.feature_bank[ind, :].cuda().permute(
                            1, 0)).argmax(dim=0)
                    target = inclusion[target_ind]
                else:
                    target = torch.zeros((ind.shape[0], ),
                                         dtype=torch.int64).cuda()
                dist.all_reduce(target)
                self.label_bank[ind] = torch.from_numpy(target.cpu().numpy())
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)

    def update_centroids_memory(self, cinds=None):
        """Update centroids memory."""
        if self.rank == 0:
            if self.debug:
                print('updating centroids ...')
            if cinds is None:
                center = self._compute_centroids()
                self.centroids.copy_(center)
            else:
                center = self._compute_centroids_ind(cinds)
                self.centroids[
                    torch.LongTensor(cinds).cuda(), :] = center.cuda()
        dist.broadcast(self.centroids, 0)

    def _partition_max_cluster(self, max_cluster):
        """Partition the largest cluster into two sub-clusters."""
        assert self.rank == 0
        max_cluster_inds = np.where(self.label_bank == max_cluster)[0]

        assert len(max_cluster_inds) >= 2
        max_cluster_features = self.feature_bank[max_cluster_inds, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception('Has nan in features.')
        kmeans_ret = self.kmeans.fit(max_cluster_features)
        sub_cluster1_ind = max_cluster_inds[kmeans_ret.labels_ == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                'Warning: kmeans partition fails, resort to random partition.')
            sub_cluster1_ind = np.random.choice(
                max_cluster_inds, len(max_cluster_inds) // 2, replace=False)
            sub_cluster2_ind = np.setdiff1d(
                max_cluster_inds, sub_cluster1_ind, assume_unique=True)
        return sub_cluster1_ind, sub_cluster2_ind

    def _redirect_empty_clusters(self, empty_clusters):
        """Re-direct empty clusters."""
        for e in empty_clusters:
            assert (self.label_bank != e).all().item(), \
                f'Cluster #{e} is not an empty cluster.'
            max_cluster = np.bincount(
                self.label_bank, minlength=self.num_classes).argmax().item()
            # gather partitioning indices
            if self.rank == 0:
                sub_cluster1_ind, sub_cluster2_ind = \
                    self._partition_max_cluster(max_cluster)
                size1 = torch.LongTensor([len(sub_cluster1_ind)]).cuda()
                size2 = torch.LongTensor([len(sub_cluster2_ind)]).cuda()
                sub_cluster1_ind_tensor = torch.from_numpy(
                    sub_cluster1_ind).long().cuda()
                sub_cluster2_ind_tensor = torch.from_numpy(
                    sub_cluster2_ind).long().cuda()
            else:
                size1 = torch.LongTensor([0]).cuda()
                size2 = torch.LongTensor([0]).cuda()
            dist.all_reduce(size1)
            dist.all_reduce(size2)
            if self.rank != 0:
                sub_cluster1_ind_tensor = torch.zeros(
                    (size1, ), dtype=torch.int64).cuda()
                sub_cluster2_ind_tensor = torch.zeros(
                    (size2, ), dtype=torch.int64).cuda()
            dist.broadcast(sub_cluster1_ind_tensor, 0)
            dist.broadcast(sub_cluster2_ind_tensor, 0)
            if self.rank != 0:
                sub_cluster1_ind = sub_cluster1_ind_tensor.cpu().numpy()
                sub_cluster2_ind = sub_cluster2_ind_tensor.cpu().numpy()

            # reassign samples in partition #2 to the empty class
            self.label_bank[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_centroids_memory([max_cluster, e])
