import logging

import numpy as np
import pandas as pd
import pyedflib

from deepsleep.data import BaseLoader
from deepsleep.data.utils import load_pyedf
from deepsleep import SPINDLE_CONSTANTS
from deepsleep import ROOT_LOGGER_STR

THIS_DATA_ROOT = SPINDLE_CONSTANTS['DATA_ROOT']
DATA_FOLDERS = SPINDLE_CONSTANTS['DATA_FOLDERS']
RAW_FOLDER = SPINDLE_CONSTANTS['RAW_FOLDER']
LABEL_FOLDER = SPINDLE_CONSTANTS['LABEL_FOLDER']
RAW_EXT = SPINDLE_CONSTANTS['RAW_EXT']
LABEL_EXT = SPINDLE_CONSTANTS['LABEL_EXT']
RAW_FILES = SPINDLE_CONSTANTS['RAW_FILES']
HDF5_NUM_CLASSES = SPINDLE_CONSTANTS['HDF5_NUM_CLASSES']

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleData(BaseLoader):
    """Load dataset from SpindleData"""

    def __init__(self, configs, preprocessing):
        """
        Args:

        """
        super().__init__(configs, preprocessing)

        self.data_folders = DATA_FOLDERS  # This is a list of folders
        self.raw_folder = RAW_FOLDER
        self.label_folder = LABEL_FOLDER
        self.raw_ext = RAW_EXT
        self.label_ext = LABEL_EXT
        self.rawfiles = RAW_FILES
        self.hdf5_num_classes = HDF5_NUM_CLASSES

        self.num_classes = configs['num_classes']
        self.agreement = configs['agreement']

        self.data_path = self.data_root / THIS_DATA_ROOT

        if not self.agreement:
            logging.warning('No agreement has been selected. '
                            'The labels from the first expert will be '
                            'taken as ground truth.')

    def read_all_data(self):
        for filename in self.rawfiles:
            for folder in self.data_folders:
                eeg_path = (self.data_path / folder / self.raw_folder
                            / (filename + self.raw_ext))
                label_path = (self.data_path / folder / self.label_folder
                              / (filename + self.label_ext))
                if not eeg_path.is_file():
                    continue

                logger.info(f'preprocessing dataset {filename} ...')

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

                labels = pd.read_csv(str(label_path),
                                     usecols=[0, 1, 2],
                                     names=['id', 'label1', 'label2'],
                                     index_col='id')

                if self.agreement:
                    data = data[labels['label1'] == labels['label2']]
                    labels = labels[labels['label1'] == labels['label2']]

                # Labels from chars to all integers
                labels = labels.replace({"label1": self._get_total_mapping()})
                labels = np.array(labels['label1'], dtype=np.int32)

                # Remove the trash labels
                data = data[labels < 6]
                labels = labels[labels < 6]

                # [A1, A2] to 'fold0' | [A3, A4] to 'fold1' | BX to 'fold2' |
                # CX to 'fold3' | DX to 'fold4'
                fold_name = ('fold' + filename[0:2]
                             .replace('A1', '0')
                             .replace('A2', '0')
                             .replace('A3', '1')
                             .replace('A4', '1')
                             .replace('B', '2')
                             .replace('C', '3')
                             .replace('D', '4')[0])
                yield fold_name, data, labels

    def transform_data(self, data, labels):
        if self.num_classes == 3:
            # labels 3, 4, 5 >> 0, 1, 2 respectively
            labels = labels % 3

        elif self.num_classes == 2:
            # Labels 0, 1, 2 >> 1
            # labels 3, 4, 5 >> 0
            labels = (labels + 3) // 3
            labels = labels % 2

        return data, labels

    def transform_meta(self):
        assert len(self.class_count) == HDF5_NUM_CLASSES
        if self.num_classes == 3:
            self.class_count = self.class_count[0:3] + self.class_count[3:6]

        elif self.num_classes == 2:
            sig_num = np.sum(self.class_count[0:3])
            noise_num = np.sum(self.class_count[3:6])
            self.class_count = np.array([noise_num, sig_num])

    @staticmethod
    def _get_4stage_mapping():
        return {"w": 0,  # WAKE
                "n": 1,  # NREM
                "r": 2,  # REM
                "1": 3, "2": 3, "3": 3, "a": 3, "'": 3, "4": 3,  # artifact
                "U": 4}  # unknown/ambiguous

    @staticmethod
    def _get_binary_mapping():
        return {"w": 1, "n": 1, "r": 1,  # regular
                "1": 0, "2": 0, "3": 0, "a": 0, "'": 0, "4": 0,  # artifact
                "U": 2}  # unknown/ambiguous

    @staticmethod
    def _get_total_mapping():
        return {"w": 0,  # WAKE
                "n": 1,  # NREM
                "r": 2,  # REM
                "1": 3, "2": 4, "3": 5,     # noise
                "a": 6, "'": 6, "4": 6, "U": 6}  # trash
