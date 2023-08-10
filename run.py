import sys
from pathlib import Path
import time
import logging
import argparse

import importlib
import mlflow
import yaml

import deepsleep.configs.constants as cnst
from deepsleep.utils import set_up_paths
from deepsleep import CONFIG_FILE
from deepsleep import LOGGER_RESULT_FILE, ROOT_LOGGER_STR

# Uncomment to fix the seed:
# import torch
# import numpy as np
# SEED = 2147483647
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


def set_handlers(config, preprocessing, is_train=True):
    if not config:
        return []

    data_handlers = []
    for dataset in config.get('datasets', []):
        # Put loader's related configs together with data related configs
        loader_configs = config.copy()
        del loader_configs['datasets']
        dataset.update(loader_configs)

        # Find the data class loader
        class_name = dataset['name']
        dataclass = get_datahandler(class_name)

        if dataset['merge_folds']:
            # Create the handler and make hdf5
            dataloader = dataclass(dataset, preprocessing)
            dataloader.make_hdf5()
            data_handlers.append(dataloader)
        else:
            folds = dataclass.set_folds_from_config(cnst.DATA_PATH, dataset)
            for fold in folds:
                # Replace the fold list with only one of the folds
                cfg = dataset.copy()
                cfg['folds'] = [fold]
                cfg['folds_path'] = None

                # Create the handler and make hdf5
                dataloader = dataclass(cfg, preprocessing)
                dataloader.make_hdf5()
                data_handlers.append(dataloader)

    if is_train:
        # Training data can not be multiple datasets. Test or validation
        # data however can be multiple dataset on which results is
        # separately returned
        assert len(data_handlers) == 1
        data_handlers = data_handlers[0]

    return data_handlers


def _find_class_using_name(folder_name, class_name):
    """Loads the class object defined in the `folder_name`. Both the class
    name and python file name must be `class_name`"""

    class_lib = importlib.import_module(folder_name)
    the_class = None
    for name, cls in class_lib.__dict__.items():
        if name.lower() == class_name.lower():
            the_class = cls
    return the_class


def get_model(model_name):
    return _find_class_using_name('deepsleep.models', model_name)


def get_datahandler(data_name):
    return _find_class_using_name('deepsleep.data', data_name)


def get_preprocessing(processing_name):
    return _find_class_using_name('deepsleep.data', processing_name)


def _setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


def setup_experiment(config):
    """Creates a new experiment folder based on name and datetime and saves
    config files there to archive. It also sets up the logger for the
    project."""

    global logger
    # Setup paths, folders, logger, configurations, etc.
    exp_name = config['name']
    exp_id = exp_name + '_' + str(time.time())
    cnst.EXPERIMENT_PATH = cnst.EXPERIMENTS_PATH / exp_id
    cnst.EXPERIMENT_PATH.mkdir()

    new_cfg_path = cnst.EXPERIMENT_PATH / CONFIG_FILE
    log_path = cnst.EXPERIMENT_PATH / LOGGER_RESULT_FILE

    with new_cfg_path.open(mode='w') as yaml_file:
        yaml.dump(config, yaml_file)
    _setup_logger(log_path, config['app']['stdlog'])

    logger.info(f"Running experiment {exp_id}")
    logger.debug(f'Experiment path created successfully: '
                 f'{cnst.EXPERIMENT_PATH}')


def run_experiments(exp_cfg):
    """Read the config file and run all the training experiments"""

    # Create folder, save configs and set up the logger
    setup_experiment(exp_cfg)

    # Setup preprocessing routine
    preprocessing_class = get_preprocessing(exp_cfg['preprocessing']['name'])
    preprocessing = preprocessing_class(exp_cfg['preprocessing'])

    # pre-process and save to hdf5. This may take a while.
    train_handlers = set_handlers(
        exp_cfg['data'].get('train', []), preprocessing, is_train=True)
    valid_handlers = set_handlers(
        exp_cfg['data'].get('validation', []), preprocessing, is_train=False)
    test_handlers = set_handlers(
        exp_cfg['data'].get('test', []), preprocessing, is_train=False)
    prediction_handlers = set_handlers(
        exp_cfg['data'].get('prediction', []), preprocessing, is_train=False)

    # Set up model
    model_class = get_model(exp_cfg['model']['name'])
    model = model_class(exp_cfg)

    # Run the model with corresponding dataset
    mlflow.set_tracking_uri(str(cnst.EXPERIMENTS_PATH / 'mlruns'))
    logger.debug('Task to run: {0}'.format(exp_cfg['task']))
    with mlflow.start_run():
        mlflow.log_param('exp_name', exp_cfg['name'])
        mlflow.log_param('model_name', exp_cfg['model']['name'])
        mlflow.log_param('prep_name', exp_cfg['preprocessing']['name'])
        mlflow.log_params(exp_cfg['model'])
        mlflow.log_params(exp_cfg['preprocessing'])

        if exp_cfg['task'] == 'train':
            # mlflow.log_params(exp_cfg['data']['train']['datasets'][0])
            mlflow.log_params(exp_cfg['training']['optimiser'])
            training_params = exp_cfg['training'].copy()
            training_params.pop('optimiser')
            mlflow.log_params(training_params)

            model.set_inputs(train=train_handlers,
                             validation=valid_handlers,
                             test=test_handlers)
            model.train()

        elif exp_cfg['task'] == 'test':
            # for dataset in exp_cfg['data']['test']['datasets']:
            #     mlflow.log_params(dataset)

            model.set_inputs(test=test_handlers)
            model.test()

        elif exp_cfg['task'] == 'prediction':
            # for dataset in exp_cfg['data']['prediction']['datasets']:
            #     mlflow.log_params(dataset)

            model.set_inputs(prediction=prediction_handlers)
            model.prediction()

        else:
            msg = (f"task {exp_cfg['task']} is not supported. Try 'test' or "
                   f"'train'")
            logger.error(msg)
            raise AttributeError(msg)


def main():
    """Read command arguments and config files and delegate the task of
    training, evaluation or prediction"""

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir",
                        type=Path,
                        help="Path in which results will of training "
                             "are/will be located")
    parser.add_argument("config",
                        type=Path,
                        help="Path to the main yaml config. "
                             "Ex: 'configs/train_spindle_ss.yaml'")
    parser.add_argument("--data_dir",
                        type=Path,
                        help="Path to the data folder.")
    parser.add_argument("--weight_dir",
                        type=Path,
                        help="Path to the weights folder")
    parser.add_argument("--tmpdir", action='store_true',
                        help="Whether to copy data to TMPDIR")

    args = parser.parse_args()

    # Read config files
    with args.config.open(mode='r') as yamlfile:
        cfgs = yaml.safe_load(yamlfile)

    # Set project wide variables
    set_up_paths(root_path=args.result_dir, data_path=args.data_dir,
                 weights_path=args.weight_dir, tmp=args.tmpdir)

    run_experiments(cfgs)


if __name__ == "__main__":
    main()
