import numpy as np
import logging

import deepsleep.data.utils as utils
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class LinusPreprocWithFiltering:
    def __init__(self, params):
        self.name = params['name']
        self.stride = params['spectrogram-stride']
        self.window = params['spectrogram-window']
        self.time_interval = params['time_interval']
        self.num_neighbors = params['num_neighbors']
        self.eeg_filtering = params['EEG-filtering']
        self.eog_filtering = params['EOG-filtering']
        self.emg_filtering = params['EMG-filtering']

        logger.debug('preprocessing routine created: \n {0}'.format(self))

    def __call__(self, all_signals, srate,
                 eeg_idx=np.array([]),
                 eog_idx=np.array([]),
                 emg_idx=np.array([])):
        logger.debug('preprocessing data ...')
        logger.debug(f'eeg indices: {eeg_idx}, eog indices: {eog_idx} and '
                     f'emg indices: {emg_idx} vs signal shape '
                     f'{all_signals.shape}')

        eegs, eogs, emgs = np.array([]), np.array([]), np.array([])
        if len(eeg_idx) > 0:
            eegs = utils.pop_eeg_filtnew(all_signals[eeg_idx, ...],
                                         srate,
                                         self.eeg_filtering['lfreq'],
                                         self.eeg_filtering['hfreq'])
        if len(eog_idx) > 0:
            eogs = utils.pop_eeg_filtnew(all_signals[eog_idx, ...],
                                         srate,
                                         self.eog_filtering['lfreq'],
                                         self.eog_filtering['hfreq'])
        if len(emg_idx) > 0:
            emgs = utils.pop_eeg_filtnew(all_signals[emg_idx, ...],
                                         srate,
                                         self.emg_filtering['lfreq'],
                                         self.emg_filtering['hfreq'])
        all_signals = [x for x in [eegs, eogs, emgs] if x.size > 0]
        all_signals = np.concatenate(all_signals, axis=0)

        spectrograms = utils.get_spectrograms(all_signals, srate, self.window,
                                              self.stride, mode='psd')

        samples_per_epoch = int(self.time_interval * srate)
        # This is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // self.stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        data = utils.make_epochs(spectrograms, num_epochs, epoch_size)

        data = utils.normalise_spectrogram_epochs(data)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data

    def __str__(self):
        return f"""Preprocessing class LinusPreprocessingWithFiltering 
        replicates the preprocessing introduced in Linus' thesis where 
        filtering is done on the raw data instead of loading the filtered data.
        parameters:
        spectrogram stride: {self.stride}
        spectrogram window: {self.window}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        eeg_filtering: lfreq={self.eeg_filtering['lfreq']}, 
                       hfreq={self.eeg_filtering['hfreq']}
        eog_filtering: lfreq={self.eog_filtering['lfreq']}, 
                       hfreq={self.eog_filtering['hfreq']}
        emg_filtering: lfreq={self.emg_filtering['lfreq']}, 
                       hfreq={self.emg_filtering['hfreq']}
        """
