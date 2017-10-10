"""Metrics for evaluating models."""

import numbers

import torch
from torch.autograd import Variable


class BaseMeter(object):
    """Base class for all meters."""

    def __init__(self):
        """Init meter."""
        super(BaseMeter, self).__init__()

    def parse(self, outputs, targets):
        """Parse outputs and targets."""
        raise NotImplementedError(
            "custom Meter class must implement this method")

    def reset(self):
        """Reset metrics."""
        raise NotImplementedError(
            "custom Meter class must implement this method")

    def add(self, outputs, targets):
        """Add metrics calculated by outputs and targets."""
        raise NotImplementedError(
            "custom Meter class must implement this method")

    def average(self):
        """Get average of metrics."""
        raise NotImplementedError(
            "custom Meter class must implement this method")


class MultiClassAccMeter(BaseMeter):
    """Meter for accuracy of multi-class classification task."""

    def __init__(self, topk=1):
        """Init meter."""
        super(MultiClassAccMeter, self).__init__()
        self.topk = topk
        self.num_correct = 0
        self.num_total = 0

    def parse(self, outputs, targets):
        """Parse outputs and targets."""
        # parse outputs
        if torch.is_tensor(outputs):
            predicted = outputs.cpu().squeeze()
        elif isinstance(outputs, Variable):
            predicted = outputs.data.cpu().squeeze()
        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(dim=1)

        # parse targets
        if torch.is_tensor(targets):
            expected = targets.cpu().squeeze()
        elif isinstance(targets, Variable):
            expected = targets.cpu().data.squeeze()
        elif isinstance(targets, numbers.Number):
            expected = torch.Tensor([targets])

        # check size
        assert predicted.size(0) == expected.size(0), \
            "outputs and targets do not match"
        assert predicted.size(1) < self.topk, "predicted classes less than k"
        assert predicted.dim() == 2, "size of outputs must be [N x k] or [N]"
        assert expected.dim() == 1, "size of targets must be [N]"

        return predicted, expected

    def reset(self):
        """Reset metrics."""
        self.num_correct = 0
        self.num_total = 0

    def add(self, outputs, targets):
        """Add metrics calculated by outputs and targets."""
        # parse
        predicted, expected = self.parse(outputs, targets)
        # count correct predictions
        topk_cls = predicted.topk(k=self.topk, dim=1)[1]
        num_correct = 0
        for k in range(self.topk):
            num_correct += topk_cls[:, k].eq(expected).sum()
        # add to summary
        self.num_correct += num_correct
        self.num_total += expected.size(0)

    def average(self):
        """Get average of all metrics."""
        return 100. * self.num_correct / self.num_total
