import numpy as np
import logging

import deepsleep.data.utils as utils
from deepsleep.data.utils import get_spectrograms as get_specs
from deepsleep.data.utils import compress_spectrograms as compress
from deepsleep.data.utils import compress_and_replicate_emg as comp_emg
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleWithEOGPreproc:
    def __init__(self, params):
        self.name = params['name']
        self.target_srate = params['target_srate']
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
        logger.debug(f'eeg indices: {eeg_idx} and emg indices: {emg_idx} '
                     f'vs signal shape: {all_signals.shape}')

        if srate != self.target_srate:
            all_signals = utils.resample(all_signals, srate, self.target_srate)
            srate = self.target_srate

        stride = self.stride
        win = srate * 2
        specs = get_specs(all_signals, srate, win, stride, mode='magnitude')

        eeg_spectrograms = compress(specs[eeg_idx, ...], srate, win,
                                    lowcutoff=self.eeg_filtering['lfreq'],
                                    highcutoff=self.eeg_filtering['hfreq'])
        eog_spectrograms = compress(specs[eog_idx, ...], srate, win,
                                    lowcutoff=self.eog_filtering['lfreq'],
                                    highcutoff=self.eog_filtering['hfreq'])
        emg_spectrogram = comp_emg(specs[emg_idx, ...], srate, win,
                                   lowcutoff=self.emg_filtering['lfreq'],
                                   highcutoff=self.emg_filtering['hfreq'],
                                   replicate=np.shape(eeg_spectrograms)[1])
        specs = [eeg_spectrograms, eog_spectrograms, emg_spectrogram]
        specs = [x for x in specs if x.size > 0]
        specs = np.concatenate(specs, axis=0)

        specs = utils.normalise_spectrograms(specs)

        samples_per_epoch = int(self.time_interval * srate)
        # This is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        data = utils.make_epochs(specs, num_epochs, epoch_size)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data

    def __str__(self):
        return f"""Preprocessing class SpindleWithEOGPreprocessing replicates 
        the exact preprocessing introduced in the paper but also preprocesses 
        EOG channels the same way as EEG channels.
        parameters:
        target_srate: {self.target_srate}
        stride: {self.stride}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        eeg_filtering: lfreq={self.eeg_filtering['lfreq']}, 
                       hfreq={self.eeg_filtering['hfreq']}
        eog_filtering: lfreq={self.eog_filtering['lfreq']}, 
                       hfreq={self.eog_filtering['hfreq']}
        emg_filtering: lfreq={self.emg_filtering['lfreq']}, 
                       hfreq={self.emg_filtering['hfreq']}
        """
