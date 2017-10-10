"""Helpful functions for datasets."""

import os

import numpy as np
import torch


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def get_inf_iterator(data_loader):
    """Inf dataset iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)


def cv2_loader(path):
    """cv2 image loader.

    This can be used to replace PIL image loader of ImageFolder dataset class.
    """
    import cv2
    return cv2.imread(path)


def split_dataset(dataset,
                  trainval_rate=0.7,
                  train_rate=0.9,
                  shuffle=True,
                  save_split=False,
                  save_path="dataset_split.pt"):
    """Split dataset into train/val/test subset."""
    all_indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(all_indices)

    # get number of images in subsets
    num_test = int(np.ceil((1 - trainval_rate) * len(all_indices)))
    num_trainval = len(all_indices) - num_test
    num_train = int(np.floor(train_rate * num_trainval))
    num_val = num_trainval - num_train

    # get indices of subsets
    test_idx = all_indices[:num_test]
    trainval_indices = [x for x in all_indices if x not in test_idx]
    train_idx = trainval_indices[num_val:]
    val_idx = trainval_indices[:num_val]

    # add to dict
    split_indices = {
        "all": all_indices,
        "trainval": trainval_indices,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    # save data split
    if save_split:
        save_dataset_split(split_indices, save_path)

    return split_indices


def save_dataset_split(split_indices, filepath):
    """Save splited dataset indices."""
    if os.path.exists(os.path.dirname(filepath)):
        torch.save(split_indices, filepath)


def load_dataset_split(filepath):
    """Load splited dataset indices."""
    split_indices = None
    if os.path.exists(filepath):
        split_indices = torch.load(filepath)
    return split_indices


def get_balanced_weights(images, num_classes):
    """Get weights for WeightedRandomSampler on classes balancing."""
    count = [0] * num_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight
