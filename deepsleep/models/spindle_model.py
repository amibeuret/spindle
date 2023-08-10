import logging

from deepsleep.models import SpindleGraph
from deepsleep.models import BaseModel
from deepsleep import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SpindleModel(BaseModel):
    def __init__(self, params, num_classes=3):
        super().__init__(params, num_classes=num_classes)

    def set_graph(self, example_set):
        # Set up the graph
        dropout_prob = (self.params['model']['dropout_prob'] if self.is_train
                        else 0.5)
        input_dim = example_set.input_dim
        self.graph = SpindleGraph(input_dim, self.num_classes, dropout_prob)
        self.graph.float().to(self.device)

        pars = sum(p.numel() for p in self.graph.parameters()
                   if p.requires_grad)
        logger.debug(f'Graph created and put to the device {self.device_name}')
        logger.debug(f'{pars} trainable parameters')

    @staticmethod
    def _get_total_mapping():
        """After reading from disk for confusion matrices"""
        return {"w": 0,  # WAKE
                "n": 1,  # NREM
                "r": 2,  # REM
                "1": 3, "2": 4, "3": 5,     # noise
                "a": 6, "'": 6, "4": 6, "U": 6}  # trash

