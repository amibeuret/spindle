import sys
from os import environ as os_environ
from pathlib import Path
import time
import logging

import deepsleep.configs.constants as cnst
from deepsleep import DATA_FOLDER, WEIGHTS_FOLDER, EXPERIMENTS_FOLDER
from deepsleep import ROOT_LOGGER_STR


def set_up_paths(root_path, data_path=None, weights_path=None, tmp=False):

    cnst.ROOT_PATH = root_path
    cnst.EXPERIMENTS_PATH = root_path / EXPERIMENTS_FOLDER
    cnst.DATA_PATH = root_path / DATA_FOLDER if not data_path else data_path
    cnst.WEIGHTS_PATH = (root_path / WEIGHTS_FOLDER if not weights_path
                         else weights_path)
    cnst.TMP_DIR = Path(os_environ['TMPDIR'] if tmp else Path(''))

    exp_id = str(time.time())
    cnst.EXPERIMENT_PATH = cnst.EXPERIMENTS_PATH / exp_id

    cnst.EXPERIMENTS_PATH.mkdir(parents=True, exist_ok=True)
    cnst.WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger_to_std():
    """Logger for debugging the code during development"""
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(f_format)
    root_logger.addHandler(handler)
