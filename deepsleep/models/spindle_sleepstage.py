import logging

from deepsleep.models import SpindleModel
from deepsleep import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleSleepStageModel(SpindleModel):
    def __init__(self, params):
        super().__init__(params, num_classes=3)

    @staticmethod
    def _map_preds(pred):
        """Before writing to disk"""
        if pred == 0:
            res = 'w'
        elif pred == 1:
            res = 'n'
        elif pred == 2:
            res = 'r'
        else:
            res = pred.cpu().numpy() - 2
        return str(res)
