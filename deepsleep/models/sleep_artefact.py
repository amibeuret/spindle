import logging

from deepsleep.models import SpindleModel
from deepsleep import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleArtefactModel(SpindleModel):
    def __init__(self, params):
        super().__init__(params, num_classes=2)

    @staticmethod
    def agreement_step(max_prob_class, labels):
        noise_intersec = ((max_prob_class == 0) & (labels == 0)).float().sum()
        noise_union = ((max_prob_class == 0) | (labels == 0)).float().sum()
        return noise_intersec, noise_union
