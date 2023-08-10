# These are not really constants. They are set once the config file is read.
ROOT_PATH = None

EXPERIMENT_PATH = None
EXPERIMENTS_PATH = None
WEIGHTS_PATH = None
DATA_PATH = None

# String constants
HDF5_KEY_INPUT_DST_STR = 'input'
HDF5_KEY_OUTPUT_DST_STR = 'output'
HDF5_KEY_OUTPUT1_DST_STR = 'output1'
HDF5_KEY_OUTPUT2_DST_STR = 'output2'
HDF5_KEY_META_DST_STR = 'metadata'

TENSOR_INPUT_KEY_STR = 'input'
TENSOR_OUTPUT_KEY_STR = 'output'

ROOT_LOGGER_STR = 'DeepSleepLogger'

# Path constants
DATA_FOLDER = 'dataset/'
EXPERIMENTS_FOLDER = 'experiments/'
WEIGHTS_FOLDER = 'weights/'
TMP_DIR = ''

MAIN_CONFIG_PATH = 'configs/train_spindle_ss.yaml'

CONFIG_FILE = 'config.yaml'
LOGGER_RESULT_FILE = 'logs.txt'
PREDICTIONS_FILE = 'predictions{0}.csv'
LABELS_FILE = 'labels{0}.csv'
CONFUSION_FIG_FILE = 'confusion-matrix'

# Data related constants
SPINDLE_CONSTANTS = {
    'DATA_ROOT': 'SpindleData/',
    'DATA_FOLDERS': ['CohortA', 'CohortB', 'CohortC', 'CohortD'],
    'RAW_FOLDER': 'recordings',
    'LABEL_FOLDER': 'scorings',
    'RAW_EXT': '.edf',
    'LABEL_EXT': '.csv',
    'HDF5_NUM_CLASSES': 6,
    'RAW_FILES': ['A1', 'A2', 'A3', 'A4',
                  'B1', 'B2', 'B3', 'B4',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                  'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
}

PRODUCTION_CONSTANTS = {
    'DATA_ROOT': 'productionData/'
}

CARO_CONSTANTS = {
    'DATA_ROOT': 'newHumanData/',
    'RAW_EXT': '.mat',
    'TRAIN_CSV': 'cv_train.csv',
    'TEST_CSV': 'cv_test.csv',
    'VAL_CSV': 'cv_val.csv',
    'max_fold': 16,
    'HDF5_NUM_CLASSES': 5,
    'CHANNEL_LIST': ['FpzA2', 'EOGR', 'EOGL', 'EMG'],
    'FILTERED_CHANNEL_LIST': ['FpzA2_filt',
                              'EOGR_filt',
                              'EOGL_filt',
                              'EMG_filt'],
    'POSSIBLE_EXPERTS': ['E1', 'E2', 'E3', 'E4', 'E5', 'CL']
}

VALKODATA_CONSTANTS = {
    'DATA_ROOT': 'Valko/data/',
    'RAW_EXT': '.mat',
    'SD_START_STR': 'SDSR_NK',
    'VP_START_STR': 'VP'
}


