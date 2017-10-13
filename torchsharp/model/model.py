"""High-level Model class."""


class BaseModel(object):
    """Base class for all other models."""

    def __init__(self):
        """Init model."""
        super(BaseModel, self).__init__()
        self.name = "BaseModel"
        self.cfg = None
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        # self.initializer = None
        self.metrics = None

    def init(self, net, cfg):
        """Init model with network and config."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def save(self, filepath):
        """Save model checkpoint."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def restore(self, filepath):
        """Restore model checkpoint."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def forward(self, input):
        """Forward network with input."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def optimize(self):
        """Optimize network."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def inference(self):
        """Inference network w/o computing gradients."""
        raise NotImplementedError(
            "custom Model class must implement this method")
