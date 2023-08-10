from .base_loader import BaseLoader

from .preproc.linus_preproc import LinusPreproc
from .preproc.linus_preproc_with_filtering import LinusPreprocWithFiltering
from .preproc.new_preproc import NewPreproc
from .preproc.normed_spindle_preproc import NormalisedSpindlePreproc
from .preproc.spindle_preproc import SpindlePreproc
from .preproc.spindle_with_eog_preproc import SpindleWithEOGPreproc

from .dataloaders.caro_data import CaroData
from .dataloaders.spindle_data import SpindleData
from .dataloaders.production_data import ProdData




