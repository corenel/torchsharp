"""Helpful functions for datasets."""


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
