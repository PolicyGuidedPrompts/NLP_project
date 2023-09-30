import time
from contextlib import contextmanager
from functools import wraps
import logging
import sys
import os
import io
import signal

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result

    return wrapper


def get_logger(log_path):
    # Ensure the directory exists
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('root')
    logger.setLevel(os.environ.get('LOGLEVEL'))

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class CaptureStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self._new_stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

    def get_output(self):
        return self._new_stdout.getvalue()


@contextmanager
def timeout(time_in_seconds):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time_in_seconds)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(_signum, _frame):
    raise TimeoutError
