"""Several useful dummy dataset."""

import os

import torch.utils.data as data


class SliceDataset(data.Dataset):
    """Slice dataset.

    That's useful when you need only a part of one dataset
    and want to use other sampler on it instead of SubsetRandomSampler.
    """

    def __init__(self, original_dataset, excerpt):
        """Init SliceDataset."""
        super(SliceDataset, self).__init__()
        self.dataset = original_dataset
        self.excerpt = excerpt
        self.imgs = [self.dataset.imgs[idx] for idx in self.excerpt]
        self.classes = self.dataset.classes

    def __getitem__(self, index):
        """Get image and target for data loader."""
        image, label = self.dataset[self.excerpt[index]]
        return image, label

    def __len__(self):
        """Return size of dataset."""
        return len(self.excerpt)


class VideoDataset(data.Dataset):
    """Dummy dataset for frames in single video."""

    def __init__(self, filepath, transform=None):
        """Init VideoDataset dataset."""
        super(VideoDataset, self).__init__()
        self.filepath = filepath
        self.num_frames = None
        self.transform = transform
        self.frames = None
        self.decode()

    def __getitem__(self, index):
        """Get frames from video."""
        frame = self.frames[index, ...]
        if self.transform is not None:
            frame = self.transform(frame)
        return index, frame

    def __len__(self):
        """Get number of the frames."""
        return self.num_frames

    def decode(self):
        """Decode frames from video."""
        import skvideo.io
        if os.path.exists(self.filepath):
            self.frames = skvideo.io.vread(self.filepath)
            # return numpy.ndarray (N x H x W x C)
            self.frames = skvideo.utils.vshape(self.frames)
            self.num_frames = self.frames.shape[0]
        else:
            raise IOError("video file doesn't exist: {}".format(self.filepath))
