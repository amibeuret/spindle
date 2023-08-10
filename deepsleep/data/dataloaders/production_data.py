import logging

import numpy as np
import pyedflib

from deepsleep.data import BaseLoader
from deepsleep.data.utils import load_pyedf
import deepsleep.configs.constants as cnst
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class ProdData(BaseLoader):
    """Load dataset from ProdData"""

    def __init__(self, configs, preprocessing):
        """
        Args:

        """
        super().__init__(configs, preprocessing)
        self.eeg_path = cnst.DATA_PATH

    def read_all_data(self):
        eeg_path = self.eeg_path
        if not eeg_path.is_file():
            logger.error(f'The given path is not an .edf file: '
                         f'{eeg_path}')

        logger.info(f'preprocessing dataset {eeg_path} ...')
        all_signals, signal_header, header = \
            pyedflib.highlevel.read_edf(str(eeg_path))

        # For production use, we replicate the eeg channel if only
        # one eeg channel exists in the original file
        if len(signal_header) == 2:
            if isinstance(all_signals, np.ndarray):
                all_signals = np.vstack(
                    (all_signals[0, ...], all_signals))
            elif isinstance(all_signals, list):
                all_signals.insert(0, all_signals[0])
            signal_header.insert(0, signal_header[0])
            logger.warning(f'The first channel of the input file is '
                           f'replicated to have two EEG channels')
        assert len(signal_header) == 3, 'Input file must have ' \
                                        'three channels'
        data = self.preprocessing(all_signals, signal_header,
                                  eeg_idx=np.array([0, 1]),
                                  eog_idx=np.array([]),
                                  emg_idx=np.array([2]))

        yield 'fold1', data, np.array([])

    def transform_meta(self):
        return

    def transform_data(self, data, labels):
        return data, labels
