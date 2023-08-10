import logging

import torch
import pandas as pd
import numpy as np
import mlflow
from hmmlearn import hmm

from deepsleep.models import BaseModel, SpindleGraph
import deepsleep.configs.constants as cnst
from deepsleep import ROOT_LOGGER_STR
from deepsleep import (TENSOR_INPUT_KEY_STR, TENSOR_OUTPUT_KEY_STR)


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindlePredictModel(BaseModel):
    def __init__(self, params):
        super().__init__(params, num_classes=6)

        # Two models in this case
        self.ad_graph = None
        self.ss_graph = None
        self.artefact_threshold = params['model']['artefact_threshold']
        if 'predictions_path' in params['model']:
            self.preds_path = params['model']['predictions_path']
        self.markov = self._set_hmm() if ('markov' in params['model'] and
                                          params['model']['markov']) else None

    def set_graph(self, exampleset):
        """Set both artefact detection and sleep staging models"""

        dropout_prob = 1
        self.ad_graph = SpindleGraph(exampleset.input_dim, 2, dropout_prob)
        self.ss_graph = SpindleGraph(exampleset.input_dim, 3, dropout_prob)
        self.ad_graph.float().to(self.device)
        self.ss_graph.float().to(self.device)
        logger.debug(f'Both graphs created and put to the device '
                     f'{self.device_name}')

    def _load_weights(self):
        """Load network's weights if specified in parameters"""""

        ad_path = str(cnst.WEIGHTS_PATH / self.params['model']['weights'][0])
        ss_path = str(cnst.WEIGHTS_PATH / self.params['model']['weights'][1])

        if self.device_name == 'cpu':
            ad_checkpoint = torch.load(ad_path, map_location='cpu')
            ss_checkpoint = torch.load(ss_path, map_location='cpu')
        else:
            ad_checkpoint = torch.load(ad_path)
            ss_checkpoint = torch.load(ss_path)

        self.ad_graph.load_state_dict(ad_checkpoint['model_state_dict'])
        self.ss_graph.load_state_dict(ss_checkpoint['model_state_dict'])
        logger.debug('Both artefact detection and sleep staging model weights '
                     'are loaded')

    def prediction(self):
        """Make predictions on prediction dataset.

        The function is used in production setting where prediction of a
        specific .edf file is saved to the given prediction's path.

        This assumes that `self.predictions_loaders` is a list of one
        element and will iterate only on the first element (DataLoader) and
        ignore if any other loaders are in the list.
        """

        thresh = self.artefact_threshold
        self.ss_graph.eval()
        self.ad_graph.eval()
        logger.info('Making predictions...')
        with torch.no_grad():
            predloader = self.prediction_loaders[0]
            ss_path = str(self.preds_path) \
                      + '_predictions_without_artifacts.csv'
            ad_path = str(self.preds_path) + '_ad_predictions_prob.csv'
            total_num = 0
            for data in predloader:
                inputs = data[TENSOR_INPUT_KEY_STR]
                ss_outputs = self.ss_graph(inputs.to(self.device))
                # C_1 >> noise, C_2 >> not noise:
                ad_outputs = self.ad_graph(inputs.to(self.device))

                ss_preds = torch.argmax(ss_outputs, dim=1)
                ss_probs = torch.exp(ss_outputs)

                # Not being noise more than threshold
                ad_preds = torch.exp(ad_outputs[:, 1]) > thresh
                ad_probs = torch.exp(ad_outputs[:, 0])

                if self.markov:
                    ss_preds = np.array(ss_probs.cpu(), dtype=np.float)
                    ss_preds = self.markov.decode(ss_preds,
                                                  algorithm="viterbi")[1]
                    ss_preds = torch.from_numpy(ss_preds)

                self._write_labels(ss_preds, total_num, ss_path)
                self._write_labels(ad_probs.cpu().numpy(), total_num, ad_path,
                                   map_it=False)

                total_num += pd.DataFrame(ss_preds,
                                          columns=['predictions']).shape[0]

            logger.info(f'Predictions of {total_num} samples successfully '
                        f'saved to file {self.preds_path}')

    def test(self):
        self.ss_graph.eval()
        self.ad_graph.eval()
        with torch.no_grad():
            num_total = 0
            total_correct = 0
            for t_i, test_loader in enumerate(self.test_loaders):
                logger.info(f'start testing on dataset {t_i} ...')
                set_num_total = 0
                set_num_correct = 0
                set_vigilance_correct = 0
                total_vig = 0
                for data in test_loader:
                    inputs = data[TENSOR_INPUT_KEY_STR].to(self.device)
                    labels = data[TENSOR_OUTPUT_KEY_STR].to(self.device)
                    ss_outputs = self.ss_graph(inputs.to(self.device))
                    # C_1 >> noise, C_2 >> not noise
                    ad_outputs = self.ad_graph(inputs.to(self.device))

                    ss_probs = torch.exp(ss_outputs)
                    preds = torch.argmax(ss_outputs, dim=1)
                    ad_preds = torch.argmax(ad_outputs, dim=1)

                    if self.markov:
                        preds = np.array(ss_probs.cpu(), dtype=np.float)
                        preds = self.markov.decode(preds,
                                                   algorithm="viterbi")[1]
                        preds = torch.from_numpy(preds).to(self.device)

                    # Separate predicted artefacts from vigilance states
                    preds[ad_preds == 0] = preds[ad_preds == 0] + 3

                    # Number of correct classified labels
                    set_num_correct += (preds == labels).float().sum()

                    # Number of correct classified vigilance state only
                    set_vigilance_correct += \
                        (preds[labels < 3] == labels[labels < 3]).float().sum()
                    total_vig += (labels < 3).float().sum()

                    # Transform each prediction to the initial labels for csv
                    p_path = self.preds_path.format(t_i)
                    l_path = self.labels_path.format(t_i)
                    self._write_labels(preds, set_num_total, p_path)
                    self._write_labels(labels, set_num_total, l_path)

                    set_num_total += labels.shape[0]
                    num_total += labels.shape[0]
                total_correct += set_num_correct

                acc = 100.0 * set_num_correct / set_num_total
                vig_acc = 100.0 * set_vigilance_correct / total_vig
                mlflow.log_metric(f'Test {t_i + 1} accuracy', acc.item())
                logger.info(f'Test accuracy for {set_num_total} data point: '
                            f'{acc:.2f} %')
                logger.info(f'Test accuracy for {total_vig} vigilance '
                            f'data point: {vig_acc:.2f} %')
                self.make_confusion_matrix(l_path, p_path, f'dataset {t_i}')
            total_acc = 100.0 * total_correct / num_total
            logger.info(f'Test accuracy for all {num_total} data points: '
                        f'{total_acc:.2f} %')

    def _set_hmm(self):
        markov = CustomHMM(n_components=3, covariance_type="full")
        markov.startprob_ = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        markov.means_ = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])
        markov.covars_ = np.tile(np.identity(3) * 0.1, (3, 1, 1))
        markov.transmat_ = np.array([[0.5, 0.5, 0.0],
                                     [0.5, 0.499, 0.001],
                                     [0.1, 0.0, 0.9]])
        return markov

    @staticmethod
    def _map_preds(pred):
        """Map the predicted label to another integer or char to be written
        to .csv file"""

        if pred == 0:
            res = 'w'
        elif pred == 1:
            res = 'n'
        elif pred == 2:
            res = 'r'
        else:
            res = pred.cpu().numpy() - 2
        return str(res)

    @staticmethod
    def _get_total_mapping():
        """Inverse operation of :meth:`._map_preds`"""

        return {"w": 0,  # WAKE
                "n": 1,  # NREM
                "r": 2,  # REM
                "1": 3, "2": 4, "3": 5,     # noise
                "a": 6, "'": 6, "4": 6, "U": 6}  # trash


class CustomHMM(hmm.GaussianHMM):
    def _compute_log_likelihood(self, X):
        return np.log(X)
