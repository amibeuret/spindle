# DeepSleep

This is a simple framework to preprocess and classify sleep stages based on 
recorded EEG/EMG/EOG data. It replicates and extends on top of the [Spindle 
method](https://sleeplearning.ethz.ch). 

## Installation
First make a new python virtual environment and then install the package using pip:

```
pip install deepsleep
```


## Spindle sleep scoring

You could use both preprocessing routines and CNN network for the classification 
demonstrated in the [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006968) 


### Spindle preprocessing routine
Below is an example of how to use the preprocessing routine:

```python
import numpy as np
import pyedflib
from deepsleep.data import SpindlePreproc
from deepsleep.utils import setup_logger_to_std


# Set the logger to print to stdout. You could skip this and ignore logs.
setup_logger_to_std()

# Adapt this for your own .edf data
eeg_file = '/path/to/eeg.edf'

# preprocessing parameters. Below are defaults used as in the paper
SPINDLE_PREPROCESSING_PARAMS = {
    'name': 'SpindlePreproc',
    'target_srate': 128,
    'spectrogram-stride': 16,
    'time_interval': 4,
    'num_neighbors': 4,
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 12},
    'EMG-filtering': {'lfreq': 2, 'hfreq': 30}
}

print("Loading EEG/EMG...")
all_signals, signal_header, header = pyedflib.highlevel.read_edf(eeg_file)

print("Preprocessing data...")
preprocessing = SpindlePreproc(SPINDLE_PREPROCESSING_PARAMS)
data = preprocessing(all_signals, signal_header, np.array([0, 1]), np.array([]), np.array([2]))

```

### Spindle CNN network

You could import the pytorch CNN network for training:
```python
from deepsleep.models import SpindleGraph

myGraph = SpindleGraph(input_dim=(3, 24, 160), nb_class=3, dropout_rate=0.5)
```

### Making prediction with Spindle

There are two ways of making predictions with our pretrained model, either by 
importing the model, preprocessing routine etc. from the library, or simply 
using the terminal and commandline: 

#### Predictions using command line


In the terminal you could simply provide the path to your `.edf` file, a path 
where you would like to save predictions `.csv`s, and the path to the directory 
where the pre-trained weights are stored as demonstrated below:

```bash
predict.py /path/to/my/AWESOME_EEG.edf --predictions /path/to/csv_folder/ --weight_dir /path/to/weights/

```

Two `.csv` files `/path/to/csv_folder/AWESOME_EEG_ad_predictions_prob.csv` and 
`/path/to/csv_folder/AWESOME_EEG_predictions_without_artifacts.csv` will be 
stored as a result.

The directory `/path/to/weights/` should contain two provided weights named 
`checkpoint_replicatePaperAD_1563189723.278988epoch3.pth` and 
`checkpoint_replicatePaperSS_1563188754.9532504epoch10.pth`

#### Predictions by importing the library

Very similar to `predict.py`, you could follow this example: (Make sure you set all the paths)

```python
from pathlib import Path

from deepsleep.data import SpindlePreproc
from deepsleep.data import ProdData
from deepsleep.models import SpindlePredictModel
from deepsleep.utils import setup_logger_to_std
from deepsleep.utils import set_up_paths


# Set parameters for preprocessing
SPINDLE_PREPROCESSING_PARAMS = {
    'name': 'SpindlePreproc',
    'target_srate': 128,
    'spectrogram-stride': 16,
    'time_interval': 4,
    'num_neighbors': 4,
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 12},
    'EMG-filtering': {'lfreq': 2, 'hfreq': 30}
}

# Set parameters for pytorch data loader
DATALOADER_PARAMS = {
    'num_workers': 4,
    'batch_size': 100,
    'do_shuffle': False,
    'batch_prefetch': 10,
    'hdf5': '',
    'folds': ['fold1']
}

# Set parameters to set up the Spindle model
MODEL_PARAMS = {
    'name': 'SpindlePredictModel',
    'artefact_threshold': 0.5,  # The probability threshold more than which the sample is considered as a noise sample
    'weights': ['checkpoint_replicatePaperAD_1563189723.278988epoch3.pth',
                'checkpoint_replicatePaperSS_1563188754.9532504epoch10.pth']
}

# Set the logger to print to stdout
setup_logger_to_std()

# Set the required paths
root_path = Path('/path/to/my/results_folder')
data_path = Path('/path/to/eeg.pdf')  # Set this to your input .edf file
weights_path = Path('/path/to/weights/')  # Set this to the folder in which weights are located
set_up_paths(root_path=root_path, data_path=data_path, weights_path=weights_path)

# Load data and preprocess
preprocessing = SpindlePreproc(SPINDLE_PREPROCESSING_PARAMS)
pred_handlers = ProdData(DATALOADER_PARAMS, preprocessing)

# Make model
params = dict({'model': MODEL_PARAMS})
params['model']['predictions_path'] = Path('/path/to/csv_folder/preds_')  #  Set this to the folder in which .csv prediction files will be saved
model = SpindlePredictModel(params)

# Set inputs and predict
model.set_inputs(prediction=[pred_handlers])
model.prediction()

```

