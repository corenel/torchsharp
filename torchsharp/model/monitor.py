"""Event monitor in training process."""


class BaseMonitor(object):
    """Base class for all other monitors."""

    def __init__(self):
        """Init monitor."""
        super(BaseMonitor, self).__init__()

    def log(self, record):
        """Writing a log in monitor."""
        raise NotImplementedError(
            "custom subclass should implement this method")
