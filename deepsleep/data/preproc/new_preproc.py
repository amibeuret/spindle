import numpy as np
import logging

import deepsleep.data.utils as utils
from deepsleep.data.utils import get_spectrograms as get_spec
from deepsleep.data.utils import compress_spectrograms as compress
from deepsleep.data.utils import compress_and_replicate_emg as comp_emg
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class NewPreproc:
    def __init__(self, params):
        self.name = params['name']
        self.stride = params['spectrogram-stride']
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
        logger.debug(f'eeg indices: {eeg_idx}, eog indices: {eog_idx} '
                     f'and emg indices: {emg_idx} vs signal shape '
                     f'{all_signals.shape}')

        stride = self.stride
        win = srate * 2  # window
        spectrograms = get_spec(all_signals, srate, win, stride, mode='psd')

        eeg_spectrograms = compress(spectrograms[eeg_idx, ...], srate, win,
                                    lowcutoff=self.eeg_filtering['lfreq'],
                                    highcutoff=self.eeg_filtering['hfreq'])
        eog_spectrograms = compress(spectrograms[eog_idx, ...], srate, win,
                                    lowcutoff=self.eog_filtering['lfreq'],
                                    highcutoff=self.eog_filtering['hfreq'])
        emg_spectrogram = comp_emg(spectrograms[emg_idx, ...], srate, win,
                                   lowcutoff=self.emg_filtering['lfreq'],
                                   highcutoff=self.emg_filtering['hfreq'],
                                   replicate=np.shape(eeg_spectrograms)[1])
        spectrograms = [eeg_spectrograms, eog_spectrograms, emg_spectrogram]
        spectrograms = [x for x in spectrograms if x.size > 0]
        spectrograms = np.concatenate(spectrograms, axis=0)

        samples_per_epoch = int(self.time_interval * srate)
        # This is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        data = utils.make_epochs(spectrograms, num_epochs, epoch_size)

        data = utils.normalise_spectrogram_epochs(data)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data

    def __str__(self):
        return f"""Preprocessing class NewPreprocessing mixes some elements of 
        Spindle and Linus preprocessing.
        parameters:
        spectrogram stride: {self.stride}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        eeg_filtering: lfreq={self.eeg_filtering['lfreq']}, 
                       hfreq={self.eeg_filtering['hfreq']}
        eog_filtering: lfreq={self.eog_filtering['lfreq']}, 
                       hfreq={self.eog_filtering['hfreq']}
        emg_filtering: lfreq={self.emg_filtering['lfreq']}, 
                       hfreq={self.emg_filtering['hfreq']}
        """
