"""Helpful transforms."""

import torch


def _is_tensor_image(img):
    """Check if the input image is a Tensor of 3 dimensions."""
    return torch.is_tensor(img) and img.ndimension() == 3


def expand_channel(img, dim, repeat):
    """Expand channel of the given Tensor image.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be expanded.
        dim (int): dimension of the channel to be repeated.
        repeat (int): repeat time.

    Returns:
        Tensor: Tensor image with expanded channels.
    """
    return torch.cat([img for cnt in range(repeat)], dim)


class ExpandChannel(object):
    """Expand channel of the given Tensor image.

    Applications:
    - MNIST: expand 1-channel images of MNIST dataset to 3-channel ones,
             usually used to train classifier with other datasets like
             MNIST-M in domain adaption.

    Args:
        dim (int): dimension of the channel to be repeated.
        repeat (int): repeat time.
    """

    def __init__(self, repeat, dim=0):
        """Init transform."""
        self.repeat = repeat
        self.dim = dim

    def __call__(self, img):
        """Call transform."""
        return expand_channel(img, self.repeat, self.dim)
