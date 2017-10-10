"""Helpful transforms for data argumentation.

Mostly modified from
https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

Example:
pre_process = transforms.Compose([Resize(cfg.image_size),
                                  RandomMirror(),
                                  ConvertFromInts(),
                                  RandomBrightness(),
                                  RandomContrast(),
                                  ConvertColor(transform="HSV"),
                                  RandomSaturation(),
                                  RandomHue(),
                                  ConvertColor(
                                      current="HSV", transform="BGR"),
                                  CV2ImageToTensor(),
                                  transforms.Normalize(
                                      mean=cfg.dataset_mean,
                                      std=cfg.dataset_std)])
"""

import random

import cv2
import numpy as np
import torch


class TensorToCV2Image(object):
    """Convert tensor to cv2 image."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, tensor):
        """Call transform."""
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class CV2ImageToTensor(object):
    """Convert cv2 image to tensor."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, cvimage):
        """Call transform."""
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class ConvertFromInts(object):
    """Convert the type of cv2 image from uint8 to float32."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, image):
        """Call transform."""
        return image.astype(np.float32)


class ConvertColor(object):
    """convert coloe space of the input cv2 image."""

    def __init__(self, current="BGR", transform="HSV"):
        """Init transform."""
        self.transform = transform
        self.current = current

    def __call__(self, img):
        """Call transform."""
        if self.current == "BGR" and self.transform == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif self.current == "RGB" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError
        return img


class Resize(object):
    """Resize cv2 image."""

    def __init__(self, size=(224, 224)):
        """Init transform."""
        self.size = size

    def __call__(self, img):
        """Call transform."""
        img = cv2.resize(img, self.size)
        return img


class RandomContrast(object):
    """Alter random contrast on the input cv2 image."""

    def __init__(self, lower=0.5, upper=1.5):
        """Init transform."""
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img


class RandomBrightness(object):
    """Alter random brightness on the input cv2 image."""

    def __init__(self, delta=32):
        """Init transform."""
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img


class RandomSaturation(object):
    """Alter random saturation on the input cv2 image."""

    def __init__(self, lower=0.5, upper=1.5):
        """Init transform."""
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            img[:, :, 1] *= random.uniform(self.lower, self.upper)

        return img


class RandomHue(object):
    """Alter random hue on the input cv2 image."""

    def __init__(self, delta=18.0):
        """Init transform."""
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img


class RandomMirror(object):
    """Random mirror image."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, img):
        """Call transform."""
        _, width, _ = img.shape
        if random.randint(0, 1):
            img = img[:, ::-1]
        return img
