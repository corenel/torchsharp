"""SliceDataset.

That's useful when you need only a part of one dataset
and want to use other sampler on it instead of SubsetRandomSampler.
"""

import torch
import torch.utils.data as data


class SliceDataset(data.Dataset):
    """Slice dataset."""

    def __init__(self, original_dataset, excerpt):
        """Init DummyDataset."""
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
