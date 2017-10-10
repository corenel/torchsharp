"""Simple logger with color."""

import logging
import logging.handlers
import os
from datetime import datetime

from color_streamhandler import ColorStreamHandler


class SimpleLogger(object):
    """Wrapper for python logger."""

    level_dict = {
        "debug": logging.INFO,
        "info": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self, name="project",
                 level=logging.INFO,
                 colorized=True,
                 save_log=True,
                 save_dir="."):
        """Init logger."""
        super(SimpleLogger, self).__init__()
        self.name = name
        self.colorized = colorized
        self.level = level
        self.save_log = save_log
        self.save_dir = save_dir

        # init logger
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)

        # init log folder
        filename = "{}-{}.log".format(self.name,
                                      datetime.now().strftime('%Y%m%d%H%M%S'))
        self.save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(self.save_dir):
            os.makedirs(os.path.dirname(self.save_path))

        # init log formatter
        # "%(asctime)s - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter("%(message)s")

        # init log handlers
        if self.colorized:
            self.console_handler = ColorStreamHandler()
        else:
            self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self._logger.addHandler(self.console_handler)

        if self.save_log:
            self.file_handler = logging.FileHandler(self.save_path)
            self.file_handler.setFormatter(self.formatter)
            self._logger.addHandler(self.file_handler)

    def debug(self, *args, **kwargs):
        """DEBUG loglevel."""
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """INFO loglevel."""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """WARNING loglevel."""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """ERROR loglevel."""
        self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        """CRITICAL loglevel."""
        self._logger.critical(*args, **kwargs)

    def set_level(self, level):
        """Set log level."""
        if level in self.level_dict:
            self._logger.setLevel(self.level_dict[level])
        else:
            self.error("level name not exists")
