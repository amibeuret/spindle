from os import mkdir
from os.path import join as pathjoin
import time
import logging
import unittest

import numpy as np
import yaml
from itertools import zip_longest

from run import get_preprocessing
from run import set_handlers
from run import _setup_logger
import deepsleep.configs.constants as cnst
from deepsleep import (EXPERIMENTS_FOLDER, DATA_FOLDER, WEIGHTS_FOLDER)
from deepsleep import LOGGER_RESULT_FILE, ROOT_LOGGER_STR
from deepsleep import (TENSOR_INPUT_KEY_STR, TENSOR_OUTPUT_KEY_STR)

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

# Constant paths
cnst.ROOT_PATH = 'resultdir/'
cnst.EXPERIMENTS_PATH = pathjoin(cnst.ROOT_PATH, EXPERIMENTS_FOLDER)
cnst.WEIGHTS_PATH = pathjoin(cnst.ROOT_PATH, WEIGHTS_FOLDER)
cnst.DATA_PATH = pathjoin(cnst.ROOT_PATH, DATA_FOLDER)
cnst.EXPERIMENT_PATH = pathjoin(cnst.EXPERIMENTS_PATH,
                                'testResults' + '_' + str(time.time()))
log_path = pathjoin(cnst.EXPERIMENT_PATH, LOGGER_RESULT_FILE)
mkdir(cnst.EXPERIMENT_PATH)

# Logger
_setup_logger(log_path, 1)

# Dataset and preprocessing configs to test
TEST_CONFIG_FILE = 'tests/hdf5_test_config.yaml'
with open(TEST_CONFIG_FILE, 'r') as yamlfile:
    configs = yaml.safe_load(yamlfile)

PREPROCESSING_NAME = configs['preprocessing']['name']
PREPROCESSING_PARAMS = configs['preprocessing']
DATASET = configs['data']


class TestHDF5(unittest.TestCase):
    def setUp(self):
        preprocessing_class = get_preprocessing(PREPROCESSING_NAME)
        preprocessing = preprocessing_class(PREPROCESSING_PARAMS)
        self.data_loader = set_handlers(DATASET, preprocessing, is_train=True)

    def test_hdf5_indexing(self):
        data_handler = self.data_loader
        _ = data_handler.get_loader()

        global_idx = 0
        for fold, data, labels in data_handler.read_all_data():
            if fold not in data_handler.folds:
                continue
            for spec, label in zip(data, labels):
                input_dict = data_handler.__getitem__(global_idx)
                spec_tensor = input_dict[TENSOR_INPUT_KEY_STR].numpy()
                label_tensor = input_dict[TENSOR_OUTPUT_KEY_STR].numpy()
                with self.subTest(global_idx):
                    self.assertTrue(
                        np.array_equal(spec.astype('float32'), spec_tensor))
                    self.assertTrue(
                        np.array_equal(label.astype('float32'), label_tensor))
                global_idx = global_idx + 1

    def test_hdf5_stability(self):
        preprocessing_class = get_preprocessing(PREPROCESSING_NAME)
        preprocessing = preprocessing_class(PREPROCESSING_PARAMS)

        sentinel = object()
        for i in range(5):
            logger.info(f'Fold {i} of stability check ...')
            dataset1 = set_handlers(DATASET, preprocessing, is_train=True)
            dataset2 = set_handlers(DATASET, preprocessing, is_train=True)

            dataset1 = dataset1.get_loader()
            dataset2 = dataset2.get_loader()
            with self.subTest(i):
                for input1, input2 in zip_longest(dataset1, dataset2,
                                                  fillvalue=sentinel):
                    spec_tensor_1 = input1[TENSOR_INPUT_KEY_STR].numpy()
                    label_tensor_1 = input1[TENSOR_OUTPUT_KEY_STR].numpy()

                    spec_tensor_2 = input2[TENSOR_INPUT_KEY_STR].numpy()
                    label_tensor_2 = input2[TENSOR_OUTPUT_KEY_STR].numpy()

                    self.assertTrue(np.all(spec_tensor_1 == spec_tensor_2))
                    self.assertTrue(np.all(label_tensor_1 == label_tensor_2))


if __name__ == '__main__':
    unittest.main()
