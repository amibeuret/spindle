import logging

from deepsleep.models import SpindleModel
from deepsleep import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleHumanSleepStageModel(SpindleModel):
    def __init__(self, params):
        super().__init__(params, num_classes=5)

    @staticmethod
    def _get_total_mapping():
        return {}
