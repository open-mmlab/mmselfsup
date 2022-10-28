# Copyright (c) Facebook, Inc. and its affiliates.

# This file is borrowed from
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py
from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def knn_eval(train_features: torch.Tensor,
             train_labels: torch.Tensor,
             test_features: torch.Tensor,
             test_labels: torch.Tensor,
             k: int,
             T: float,
             num_classes: int = 1000) -> Tuple[float, float]:
    """Compute accuracy of knn classifier predictions.

    Args:
        train_features (Tensor): Extracted features in the training set.
        train_labels (Tensor): Labels in the training set.
        test_features (Tensor): Extracted features in the testing set.
        test_labels (Tensor): Labels in the testing set.
        k (int): Number of NN to use.
        T (float): Temperature used in the voting coefficient.
        num_classes (int): Number of classes. Defaults to 1000.

    Returns:
        Tuple[float, float]: The top1 and top5 accuracy.
    """
    top1, top5, total = 0.0, 0.0, 0
    train_features = nn.functional.normalize(train_features, dim=1)
    test_features = nn.functional.normalize(test_features, dim=1)
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    # split all test images into several chunks to prevent out-of-memory
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx:min((idx +
                                          imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx:min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(
            5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5
