from collections.abc import Callable
import numpy as np
import sys
import os
import datetime
import logging
import logging.handlers
from pathlib import Path
from functools import wraps

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(*args, **kwargs):
        """
        this lets avoid adding Jupyter dependencies in production code
        but keeps functionality for local debugging without changing another source code
        """
        # return print(*args, **kwargs)
        pass


LOGGER_NAME = 'uplift_model_logger'
LOGGING_FILE_BASE_NAME = 'python-modules'
LOGGING_FILE_EXTENSION = '.log'
LOGGING_FILE_SIZE_BYTES = 1024 * 1024
LOGGING_TOTAL_FILES = 10


def log_decorator(logger_name: str = 'decorator_logger', log_args: bool = False):
    def logging_decorator(func: Callable):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            if log_args:
                s = f"{func.__name__}, args: {args}, kwargs: {kwargs}"
            else:
                s = f"{func.__name__}"
            logger.info(f"entering {s}")
            start_time = datetime.datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"exited {s}, execution took {duration} seconds")
            return result
        return wrapped_function
    return logging_decorator


class OneLineExceptionFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super(OneLineExceptionFormatter, self).formatException(exc_info)
        return repr(result)

    def format(self, record):
        s = super(OneLineExceptionFormatter, self).format(record)
        if record.exc_text:
            s = s.replace('\n', '') + '|'
        return s


def setup_logging(
    log_base_path: str = None,
    name: str = LOGGER_NAME,
    level: int = logging.DEBUG,
    to_stdout: bool = False,
):
    """
    sets up logging configuration
    """
    logger = logging.getLogger(name)
    formatter = OneLineExceptionFormatter(
        '%(asctime)s,%(msecs)03d ' + ' %(levelname)s %(filename)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )
    logger.handlers.clear()
    if level >= 0:
        logger.setLevel(level)
        if to_stdout:
            file_handler = logging.StreamHandler(sys.stdout)
        else:
            # include process id to ensure a single process writing into its own log file
            process_id = str(os.getpid())
            # now creating filepath here and folder path is provided from outside
            logs_filepath = Path(log_base_path) / (LOGGING_FILE_BASE_NAME + '_' + process_id + LOGGING_FILE_EXTENSION)
            file_handler = logging.handlers.RotatingFileHandler(
                filename = logs_filepath, maxBytes = LOGGING_FILE_SIZE_BYTES, backupCount = LOGGING_TOTAL_FILES,
            )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False


def set_seed(seed: int = None):
    if seed is not None:
        np.random.seed(seed)