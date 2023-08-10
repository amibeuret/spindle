import numpy as np
from scipy import signal
import mne
import pyedflib


def load_edf(eeg_file):
    """This function uses mne library.
    WARNING: The function has strange behaviour, please don't use"""
    data = mne.io.read_raw_edf(eeg_file)

    sfreq = data.info['sfreq']
    signals = data.get_data()
    return signals, int(sfreq)


def take_majority(labels, axis):
    u, indices = np.unique(labels, return_inverse=True)

    labels = np.apply_along_axis(np.bincount,
                                 axis,
                                 indices.reshape(labels.shape),
                                 None, np.max(indices) + 1)
    labels = u[np.argmax(labels, axis=axis)]
    return labels


def load_pyedf(eeg_file):
    f = pyedflib.EdfReader(eeg_file)
    n = f.signals_in_file
    signals = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        signals[i, :] = f.readSignal(i)
    srate = f.getSignalHeaders()[0]['sample_rate']
    return signals, srate


def slice_resample(eegs, from_srate, to_srate):
    new_len = int(len(eegs[0]) - (len(eegs[0]) % (from_srate * 4)))
    n_channels = 3
    lslice = 2 ** 15 * 100
    num_wholes = new_len // lslice
    remaining = new_len % lslice
    channels_new = [None] * n_channels
    for c in range(n_channels):
        channels_new[c] = []
        for i in range(num_wholes):
            channels_new[c].extend(
                list(signal.resample(eegs[c][i * lslice:(i + 1) * lslice],
                                     lslice // from_srate * to_srate)))
        # resample remaining part
        channels_new[c].extend(list(signal.resample(
            eegs[c][num_wholes * lslice:],
            remaining // from_srate * to_srate)))
    eegs = np.array(channels_new)
    return eegs


def resample(eegs, from_srate, to_srate):
    n_eeg, eeg_length = np.shape(eegs)
    resampled_eegs = \
        np.zeros((n_eeg, int(eeg_length * (to_srate / from_srate))))
    for i, eeg in enumerate(eegs):
        resampled_eegs[i, :] = signal.resample(
            eeg, int(len(eeg)/from_srate*to_srate))

    return resampled_eegs


def pop_eeg_filtnew(signal, srate, l_freq, h_freq):

    return mne.filter.filter_data(
        signal, srate, l_freq, h_freq,
        picks=None, filter_length='auto', l_trans_bandwidth='auto',
        h_trans_bandwidth='auto', n_jobs='cuda', method='fir',
        iir_params=None, copy=True,phase='zero', fir_window='hamming',
        fir_design='firwin', pad='reflect_limited', verbose=False)


def get_spectrograms(eegs, srate, window, stride, mode='magnitude'):
    padding = window // 2 - stride // 2

    n_eeg, eeg_length = np.shape(eegs)
    spectrograms = np.zeros((n_eeg, window // 2 + 1, eeg_length // stride))
    for i, eeg in enumerate(eegs):
        padded_eeg = np.pad(eeg, pad_width=padding, mode='edge')
        f, t, eeg_spectrogram = signal.spectrogram(padded_eeg, fs=srate,
                                                   nperseg=window,
                                                   noverlap=window - stride,
                                                   scaling='density',
                                                   mode=mode)
        eeg_spectrogram = (eeg_spectrogram / 2) ** 2 / window
        spectrograms[i, ...] = np.log(eeg_spectrogram + 0.00000001)
    return spectrograms


def get_spectrograms_vec(eegs, srate, stride, spec_mode='magnitude'):
    """Initially I thought this would be faster than the previous function,
    but not necessarily"""
    window = srate * 2
    padding = window // 2 - stride // 2

    padded_eeg = np.pad(eegs,
                        pad_width=((0, 0), (padding, padding)),
                        mode='edge')
    f, t, eeg_spectrogram = signal.spectrogram(padded_eeg, fs=srate,
                                               nperseg=window,
                                               noverlap=window - stride,
                                               scaling='density',
                                               mode=spec_mode)
    eeg_spectrogram = (eeg_spectrogram / 2) ** 2 / window
    spectrograms = np.log(eeg_spectrogram + 0.00000001)
    return spectrograms


def compress_spectrograms(eeg_specs, srate, window,
                          lowcutoff=0.5, highcutoff=12):
    nu = np.fft.rfftfreq(window, 1.0 / srate)

    new_specs = []
    for spec in eeg_specs:
        new_specs.append(spec[np.logical_and(nu >= lowcutoff,
                                             nu <= highcutoff), :])

    new_specs = np.array(new_specs)
    if len(new_specs.shape) < 2:
        new_specs = new_specs[np.newaxis, ...]
    return np.array(new_specs)


def compress_and_replicate_emg(emg_specs, srate, window,
                               lowcutoff=2, highcutoff=30,
                               replicate=1):
    nu = np.fft.rfftfreq(window, 1.0 / srate)

    new_emg_specs = np.zeros(
        (emg_specs.shape[0], replicate, emg_specs.shape[2]))
    for idx, emg_spec in enumerate(emg_specs):
        new_emg_specs[idx, ...] = np.asarray(
            [np.mean(emg_spec[np.logical_and(
                nu >= lowcutoff, nu <= highcutoff), :], 0) for _ in
             range(replicate)])

    return new_emg_specs


def replicate_emg(emg_specs, replicate=1):
    new_emg_specs = np.zeros((emg_specs.shape[0],
                              replicate,
                              emg_specs.shape[2]))
    for idx, emg_spec in enumerate(emg_specs):
        new_emg_specs[idx, ...] = np.asarray([np.mean(emg_spec, 0)
                                              for _ in range(replicate)])

    return new_emg_specs


def normalise_spectrograms(eeg_specs):
    new_eeg_specs = np.zeros(np.shape(eeg_specs))
    for j, spec in enumerate(eeg_specs):
        for i in range(np.shape(spec)[0]):
            new_eeg_specs[j, i, :] = \
                (spec[i] - np.mean(spec[i])) / np.std(spec[i])
    return new_eeg_specs


def normalise_spectrogram_epochs(eeg_specs):
    """Exactly as the previous normalisation, only can be applied after
    epochs are separated"""
    mean = np.mean(eeg_specs, axis=(0, 3), keepdims=True)
    std = np.std(eeg_specs, axis=(0, 3), keepdims=True)
    norm = (eeg_specs - mean) / (std + 1e-10)
    return norm


def add_neighbors(spectrograms, num_neighbors):
    length = np.shape(spectrograms)[3]
    aug_data = np.zeros((np.shape(spectrograms)[0],
                         np.shape(spectrograms)[1],
                         np.shape(spectrograms)[2],
                         length * (num_neighbors + 1)))
    for i in range(np.shape(aug_data)[0]):
        for j in range(num_neighbors + 1):
            aug_data[i, :, :, j * length:(j + 1) * length] = \
                spectrograms[int((i + j - int(num_neighbors / 2)) %
                                 len(spectrograms)), :, :, :]

    return aug_data


def make_epochs(spectrograms, num_epochs, epoch_size):
    spectrum_size = np.shape(spectrograms)[1]
    data = np.ndarray(shape=(num_epochs,
                             len(spectrograms),
                             spectrum_size,
                             epoch_size))
    for i in range(num_epochs):
        for j in range(len(spectrograms)):
            data[i, j, :, :] = \
                spectrograms[j][:, i * epoch_size:(i + 1) * epoch_size]
    return data
