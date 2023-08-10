#!/usr/bin/env python
"""
This file is the gateway of the sleepserver to make predictions. Instead of
the more general `run.py`, this file takes only commandline arguments and
does not require a config file. As a result, predictions are not much
configurable and use preset default values defined below. If unsure, please
use `run.py`.
"""
from pathlib import Path
import argparse
import logging

from deepsleep.data import SpindlePreproc
from deepsleep.data import ProdData
from deepsleep.models import SpindlePredictModel
from deepsleep.utils import set_up_paths, setup_logger_to_std
from deepsleep import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


SPINDLE_PREPROCESSING_PARAMS = {
    'name': 'SpindlePreproc',
    'target_srate': 128,
    'spectrogram-stride': 16,
    'time_interval': 4,
    'num_neighbors': 4,
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 12},
    'EMG-filtering': {'lfreq': 2, 'hfreq': 30}
}

DATALOADER_PARAMS = {
    'num_workers': 4,
    'batch_size': 100,
    'do_shuffle': False,
    'batch_prefetch': 100,
    'hdf5': '',
    'folds': ['fold1']
}

MODEL_PARAMS = {
    'name': 'SpindlePredictModel',
    'weights': ['checkpoint_replicatePaperAD_1563189723.278988epoch3.pth',
                'checkpoint_replicatePaperSS_1563188754.9532504epoch10.pth']
}


def predict(arg):

    # Load data and preprocess
    preprocessing = SpindlePreproc(SPINDLE_PREPROCESSING_PARAMS)
    pred_handlers = ProdData(DATALOADER_PARAMS, preprocessing)

    # Make model
    params = dict({'model': MODEL_PARAMS})
    params['model']['predictions_path'] = arg.predictions
    params['model']['artefact_threshold'] = arg.artefact_threshold
    params['model']['markov'] = arg.markov
    model = SpindlePredictModel(params)

    # Set inputs and predict
    model.set_inputs(prediction=[pred_handlers])
    model.prediction()


if __name__ == "__main__":
    setup_logger_to_std()

    parser = argparse.ArgumentParser(description='Makes predictions for the '
                                                 'sleep server')

    parser.add_argument(
        "eegfile",
        help="Path to the eegfile on which predictions are made",
        type=Path)

    parser.add_argument(
        "-t", "--artefact_threshold",
        help="The probability threshold more than which the sample is "
             "considered as a noise sample",
        type=float,
        default=0.5)

    parser.add_argument(
        "--markov",
        action='store_true',
        help="Whether to use HMM or not")

    parser.add_argument(
        "--predictions",
        help="The folder in which .csv prediction files will be saved.",
        type=Path,
        default=None)

    parser.add_argument(
        "--weight_dir",
        help="Path to the weights folder",
        type=Path)

    args = parser.parse_args()
    if args.predictions is None:
        args.predictions = args.eegfile.parent

    args.predictions /= args.eegfile.name.replace(args.eegfile.suffix, '')
    root_path = args.predictions.parent
    set_up_paths(root_path=root_path, data_path=args.eegfile,
                 weights_path=args.weight_dir, tmp=False)

    predict(args)
