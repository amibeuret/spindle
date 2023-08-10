import logging

import numpy as np
import scipy.io as sio

from deepsleep.data import BaseLoader
from deepsleep.data.utils import take_majority
from deepsleep import CARO_CONSTANTS
from deepsleep import ROOT_LOGGER_STR

THIS_DATA_ROOT = CARO_CONSTANTS['DATA_ROOT']
RAW_EXT = CARO_CONSTANTS['RAW_EXT']
CHANNEL_LIST = CARO_CONSTANTS['CHANNEL_LIST']
FILTERED_CHANNEL_LIST = CARO_CONSTANTS['FILTERED_CHANNEL_LIST']
POSSIBLE_EXPERTS = CARO_CONSTANTS['POSSIBLE_EXPERTS']
HDF5_NUM_CLASSES = CARO_CONSTANTS['HDF5_NUM_CLASSES']

TRAIN_CSV = CARO_CONSTANTS['TRAIN_CSV']
TEST_CSV = CARO_CONSTANTS['TEST_CSV']
VAL_CSV = CARO_CONSTANTS['VAL_CSV']

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class CaroData(BaseLoader):
    """Load dataset from Caro"""

    def __init__(self, configs, preprocessing):
        """
        Args:

        """
        super().__init__(configs, preprocessing)

        self.raw_ext = RAW_EXT
        self.hdf5_num_classes = HDF5_NUM_CLASSES
        self.artefact_threshold = configs['artefact_threshold']

        self.raw = configs['raw']

        self.data_path = self.data_root / THIS_DATA_ROOT

    def read_all_data(self):
        all_files = self.data_path.iterdir()
        all_files = [file for file in all_files
                     if str(file).endswith(self.raw_ext)]
        for file in all_files:
            eeg_path = self.data_path / file

            logger.info(f'Loading {file} ...')
            try:
                mat_data = sio.loadmat(eeg_path)
            except Exception as e:
                logger.error(f"{file} could not be loaded: {e}")
                continue

            s_rate = int(mat_data['sampling_rate'][0][0])
            epoch_score_len = int(mat_data['epoch_size_scoring_sec'][0][0])

            labels = []
            artefacts = []
            for expert in POSSIBLE_EXPERTS:
                if f'sleepStage_score_{expert}' in mat_data:
                    scores = mat_data[f'sleepStage_score_{expert}'][0]

                    # nb of samples per epoch marked as artefact: between 0-5
                    arts = np.sum(mat_data[f'artfact_per4s_{expert}'], axis=1)

                    labels.append(scores)
                    artefacts.append(arts)

            # np.mean in case of several experts
            artefacts = np.mean(artefacts, axis=0)
            labels = take_majority(np.array(labels), axis=0)

            # Take relevant channels and their indices
            all_signals = []
            eeg_idx = np.array([], dtype=np.int8)
            eog_idx = np.array([], dtype=np.int8)
            emg_idx = np.array([], dtype=np.int8)
            channel_list = CHANNEL_LIST if self.raw else FILTERED_CHANNEL_LIST
            for channel_name in channel_list:
                if channel_name in mat_data:
                    channel = mat_data[channel_name][0]
                    cutoff_idx = channel.shape[0] % (epoch_score_len * s_rate)
                    all_signals.append(channel[:-cutoff_idx])

                    if channel_name.startswith('FpzA2'):
                        eeg_idx = np.append(eeg_idx, len(all_signals) - 1)
                    elif channel_name.startswith('EOG'):
                        eog_idx = np.append(eog_idx, len(all_signals) - 1)
                    elif channel_name.startswith('EMG'):
                        emg_idx = np.append(emg_idx, len(all_signals) - 1)

            logger.info(f'preprocessing dataset {file} ...')
            all_signals = np.array(all_signals)
            data = self.preprocessing(all_signals, s_rate,
                                      eeg_idx=eeg_idx,
                                      eog_idx=eog_idx,
                                      emg_idx=emg_idx)
            logger.debug(f'Data size after preprocessing and making epochs '
                         f'{data.shape}')

            labels = labels[artefacts <= self.artefact_threshold, ...]
            data = data[artefacts <= self.artefact_threshold, ...]
            logger.debug(f'{np.sum(artefacts > self.artefact_threshold)} '
                         f'artefact samples removed with threshold '
                         f'{self.artefact_threshold}. '
                         f'Final data size: {data.shape}')

            labels[labels == 5] = 4
            logger.debug('Changed labels 5 to 4.')

            data = data[labels < 5]
            labels = labels[labels < 5]
            logger.debug(f'Taking only labels that are less than 5. '
                         f'Final size {data.shape}')

            if len(data) < 1 or len(labels) < 1:
                errmsg = f'Empty file is not smart: {eeg_path}?'
                logging.debug(errmsg)
                continue

            fold = str(file.with_suffix('').name)
            yield fold, data, labels

    def transform_data(self, data, labels):
        return data, labels

    def transform_meta(self):
        pass
