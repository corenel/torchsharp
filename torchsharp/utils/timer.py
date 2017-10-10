"""Timer."""

import time


class Timer(object):
    """Timer class."""

    def __init__(self, time_target=0):
        """Init timer."""
        super(Timer, self).__init__()
        self._start = time.time()
        self.time_target = time_target

    def restart(self):
        """Restart timer."""
        self._start = time.time()

    def get_elapsed(self):
        """Get elapsed time since start as secs."""
        return time.time() - self._start

    def elapsed(self):
        """Print elapsed time since start as secs."""
        print("elapsed time: {:.5f}s".format(time.time() - self._start))
        self.restart()

    def finished(self):
        """Return if time_target sec has passed."""
        return self.elapsed() >= self.time_target

    def sleep(self, duration):
        """Sleep for a while."""
        time.sleep(duration)

    def leftshift(self, time):
        """Leftshift timer."""
        self._start -= time
