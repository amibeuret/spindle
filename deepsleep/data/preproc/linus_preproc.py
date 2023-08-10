import numpy as np
import logging

import deepsleep.data.utils as utils
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class LinusPreproc:
    def __init__(self, params):
        self.name = params['name']
        self.stride = params['spectrogram-stride']
        self.window = params['spectrogram-window']
        self.time_interval = params['time_interval']
        self.num_neighbors = params['num_neighbors']

        logger.debug('preprocessing routine created: \n {0}'.format(self))

    def __call__(self, all_signals, srate,
                 eeg_idx=np.array([]),
                 eog_idx=np.array([]),
                 emg_idx=np.array([])):

        logger.debug('preprocessing data ...')
        logger.debug(f'eeg indices: {eeg_idx}, eog indices: {eog_idx} and '
                     f'emg indices: {emg_idx} vs signal shape '
                     f'{all_signals.shape}')
        spectrograms = utils.get_spectrograms(all_signals, srate, self.window,
                                              self.stride, mode='psd')

        samples_per_epoch = int(self.time_interval * srate)
        # epoch_size is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // self.stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        data = utils.make_epochs(spectrograms, num_epochs, epoch_size)

        data = utils.normalise_spectrogram_epochs(data)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data

    def __str__(self):
        return f"""
        Preprocessing class LinusPreprocessing replicates the exact 
        preprocessing introduced in Linus' thesis.
        parameters:
        spectrogram stride: {self.stride}
        spectrogram window: {self.window}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        """
