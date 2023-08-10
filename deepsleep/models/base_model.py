import logging
import time

from torch.cuda import is_available as has_cuda
from torch.cuda import get_device_name as cuda_name
import torch
import torch.nn as nn
from torch import optim
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import mlflow

from abc import abstractmethod
from deepsleep.models.utils import plot_confusion_mat
from deepsleep.models.utils import RAdam
import deepsleep.configs.constants as cnst
from deepsleep import (TENSOR_INPUT_KEY_STR, TENSOR_OUTPUT_KEY_STR)
from deepsleep import ROOT_LOGGER_STR
from deepsleep import (PREDICTIONS_FILE, LABELS_FILE, CONFUSION_FIG_FILE)

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class BaseModel:
    """Base model for training, evaluating, testing and making predictions

    In addition to the parameters below, paths attributes of
    :const:`deepsleep.configs.constants` must be set (see
    :func:`deepsleep.utils.set_up_paths()`)

    :param params: dictionary containing parameters required in this class
    :type config: dict
    :param num_classes: Number of classes for classification. This is the
        last layer of the :class:`torch.nn.Module` network
    :type num_classes: int
    :param: kwargs: Additional arguments
        This is given entirely to :meth:`~set_inputs()`

    :Example:

        >>> model = SpindlePredictModel(params)
        >>> model.set_inputs(prediction=[pred_handlers])
        >>> model.prediction()
    """
    def __init__(self, params, num_classes=3, **kwargs):

        # Get arguments
        self.params = params
        self.num_classes = num_classes

        self.start_epoch = 0
        self.n_best_accs = None
        self.is_best = False

        self.device_name = 'cuda' if has_cuda() else 'cpu'
        self.device = torch.device(self.device_name)

        cnst.EXPERIMENT_PATH.mkdir(exist_ok=True)
        self.preds_path = str(cnst.EXPERIMENT_PATH / PREDICTIONS_FILE)
        self.labels_path = str(cnst.EXPERIMENT_PATH / LABELS_FILE)

        self.is_train = ('training' in self.params
                         and self.params['training'] is not None)

        (logger.info(f'cuda device: {cuda_name(self.device)}') if has_cuda()
         else logger.info(f'using only {self.device_name}'))

        # Parameters below will be set once data inputs are set here or in
        # `set_inputs`
        self.train_set = None
        self.valid_sets = None
        self.test_sets = None
        self.prediction_sets = None

        self.train_loader = None
        self.valid_loaders = None
        self.test_loaders = None
        self.prediction_loaders = None

        self.graph = None
        self.class_weights = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Parameters only in case of training
        self.resume_checkpoint_path = None
        self.save_model = False
        self.n_best_models = 0
        if self.is_train:
            self.save_model = self.params['training']['save_model']
            self.n_best_models = self.params['training']['n_best_models']

            if self.params['training']['resume']:
                self.resume_checkpoint_path = \
                    self.params['training']['checkpoint']

        if kwargs:
            self.set_inputs(**kwargs)

        logger.debug('Model created: \n {0}'.format(self))

    def set_inputs(self, **kwargs):
        """ Set the input data

        In order to train, validate or test, the model needs an input
        dataset of type :class:`deepsleep.data.BaseLoader`. This method must
        be called before any call to other public methods to first set the
        input data and therefore initialise the model's network graph,
        optimiser, scheduler etc. which all depend on information related to
        the input data such as input dimensions. The same `kwargs` can be
        given in the constructor. In that case, no call to this method is
        required anymore.

        `kwargs` can set any of the following options: `train`, `validation`,
        `test` and `prediction`. Each of those datasets are used in
        corresponding public methods. Except the training dataset,
        all others could map to a list of :class:`deepsleep.data.BaseLoader`
        where test, prediction or validation are performed on multiple
        datasets. Each of these options are optional; however, at least one of
        them must be given.

        :param kwargs: dictionary containing datasets for training, testing,
            validating and prediction
        """
        example_set = None
        train_example_set = None
        for key, value in kwargs.items():
            if key == 'train':
                self.train_set = value
                self.train_loader = self.train_set.get_loader()
                train_example_set = value
            elif key == 'validation' and value:
                self.valid_sets = value
                self.valid_loaders = self._set_loaders(value)
                example_set = value[0]
            elif key == 'test' and value:
                self.test_sets = value
                self.test_loaders = self._set_loaders(value)
                example_set = value[0]
            elif key == 'prediction' and value:
                self.prediction_sets = value
                self.prediction_loaders = self._set_loaders(value)
                example_set = value[0]
            else:
                raise ValueError("Wrong type of dataset")

        ex_set = (train_example_set if train_example_set is not None
                  else example_set)

        # Once we have information on input dataset, set up the torch.nn graph
        self.set_graph(ex_set)

        # Load network's weight if necessary
        self._load_weights()

        # Return if it is not training mode. No need to set up the rest.
        if not self.is_train:
            logger.debug('No training. Optimiser and criterion set to None')
            return

        # In case of training, set up the necessary attributes
        self._set_optimiser()
        self._set_scheduler()
        self._get_class_weights(example_set)
        self._set_criterion()
        self._resume_state()

    @abstractmethod
    def set_graph(self, example_set):
        """Set a torch.nn module as the model's graph

        This method must be implemented to assign a torch.nn module to the
        self.graph depending on the network. This method is called from the
        :meth:`.set_inputs()` once training, validation or test data are
        assigned. This is because `example_set` is required to get
        information such as the input dimension to create the torch.nn graph.

        :param: example_set: an instance of
            :class:`~deepsleep.data.BaseLoader`, preferably the training
            example
        :type example_set: :class:`~deepsleep.data.BaseLoader`
        """

        pass

    @staticmethod
    def _map_preds(pred):
        """Map the predicted label to another integer or char to be written
        to .csv file

        This is called just before writing the prediction's list to .csv
        file. If labels different from 0, 1, 2, ... are desired, this must
        be overridden to change the prediction's list to anything desired.

        :param pred: integer prediction predicted by the nn.torch model
        :type pred: int
        :return: an integer or character representing the prediction
        :rtype: int or str
        """
        return str(pred.item())

    @staticmethod
    def _get_total_mapping():
        """Inverse operation of :meth:`._map_preds`

        This is called after reading back predictions from disk to create
        confusion matrices. This must be overridden if :meth:`_map_preds`
        is also overridden.

        :return: A dictionary mapping each prediction in .csv to an integer
        :rtype: dict

        .. seealso:: :class:`deepsleep.models.SpindleModel`
        """

        return {}

    @staticmethod
    def agreement_step(higherprobclass, labels):
        """Returns the agreement between predictions and true labels

        Agreements between predictions and labels are different based on
        different scenarios. This default implementation is the normal case
        where total number of correct predictions are divided by the total
        number of predictions. The function returns two values, one the
        nominator and the other denominator in order to generalise for all
        possible cases. This must be overridden if other kind of agreement
        is desired.

        :param higherprobclass: the predicted class
        :type higherprobclass: :class:`torch.Tensor`
        :param labels: ground truth labels
        :type labels: :class:`torch.Tensor`
        :return: tuple of tensors, nominator and denominator to compute the
            agreement
        """
        num_correct = (higherprobclass == labels).float().sum()
        num_total = labels.shape[0]
        return num_correct, num_total

    def train(self):
        """Start or resume training on training dataset"""

        logger.info(f'start training with {self}')
        params = self.params['training']
        sched_str = params['optimiser']['scheduler']
        global_step = 0
        self.n_best_accs = np.zeros(self.n_best_models)
        for epoch in range(self.start_epoch + 1, params['num_epoch'] + 1):
            logger.info(f'Epoch {epoch}')
            self.graph.train()
            for i, data in enumerate(self.train_loader, 0):
                # Get data
                inputs = data[TENSOR_INPUT_KEY_STR].to(self.device)
                labels = data[TENSOR_OUTPUT_KEY_STR].to(self.device)
                self.optimizer.zero_grad()

                # Forward pass and optimisation
                start_time = time.time()
                outputs = self.graph(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                duration = time.time() - start_time

                # Metrics
                max_prob_class = torch.argmax(outputs, dim=1)
                step_num_correct, step_num_total = \
                    self.agreement_step(max_prob_class, labels)
                accuracy = 100.0 * step_num_correct / step_num_total

                # Logs
                mlflow.log_metric('train batch acc', accuracy.item())
                if global_step % params['print_steps'] == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f'training accuracy at step {i}, '
                                f'epoch {epoch} of training: {accuracy:.2f}%, '
                                f'duration: {duration:.3f} sec/batchm, '
                                f'loss: {loss:.3f} '
                                f'lr: {lr}')

                if self.scheduler is not None and sched_str == 'cycliclr':
                    self.scheduler.step()
                global_step = global_step + 1

            self.evaluation()
            self._save_state(epoch=epoch)
        self.test()

    def evaluation(self):
        """Start validation on validation dataset"""
        if not self.valid_loaders:
            return

        self.graph.eval()
        with torch.no_grad():
            for v_i, valid_loader in enumerate(self.valid_loaders):
                logger.info(f'start validation on dataset {v_i} ...')
                sched_str = self.params['training']['optimiser']['scheduler']
                valid_loss = 0.0
                num_dom = 0
                num_denom = 0
                num_total = 0
                for data in valid_loader:
                    # Get data
                    inputs = data[TENSOR_INPUT_KEY_STR].to(self.device)
                    labels = data[TENSOR_OUTPUT_KEY_STR].to(self.device)

                    # Forward pass and loss
                    outputs = self.graph(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()

                    # More metrics
                    higherprobclass = torch.argmax(outputs, dim=1)
                    step_num_correct, step_num_total = \
                        self.agreement_step(higherprobclass, labels)
                    num_dom += step_num_correct
                    num_denom += step_num_total
                    num_total += labels.shape[0]

                acc = 100.0 * num_dom / num_denom
                valid_loss = valid_loss / num_total

                # Best model
                if np.any(acc.item() > self.n_best_accs):
                    idx = np.where(acc.item() > self.n_best_accs)[-1]
                    self.n_best_accs[idx] = acc.item()
                    self.is_best = True
                else:
                    self.is_best = False

                if self.scheduler is not None and sched_str == 'plateau':
                    self.scheduler.step(valid_loss)

                # Logs
                mlflow.log_metric(f'validation {v_i + 1} accuracy', acc.item())
                mlflow.log_metric(f'validation {v_i + 1} loss', valid_loss)
                logger.debug(f'Validation loss for {num_total} data point: '
                             f'{valid_loss:.2f}')
                logger.info(f'validation accuracy for {num_total} data point: '
                            f'{acc:.2f} %')

    def test(self):
        """Start test on test dataset"""
        if not self.test_loaders:
            return

        self.graph.eval()
        with torch.no_grad():
            for t_i, testloader in enumerate(self.test_loaders):
                folds = '-'.join(self.test_sets[t_i].folds)
                logger.info(f'start testing on dataset {t_i} '
                            f'with folds {folds}')
                num_dom = 0
                num_denom = 0
                num_total = 0
                for data in testloader:
                    inputs = data[TENSOR_INPUT_KEY_STR].to(self.device)
                    labels = data[TENSOR_OUTPUT_KEY_STR].to(self.device)

                    outputs = self.graph(inputs)
                    higherprobclass = torch.argmax(outputs, dim=1)

                    step_num_correct, step_num_total = \
                        self.agreement_step(higherprobclass, labels)
                    num_dom += step_num_correct
                    num_denom += step_num_total
                    num_total += labels.shape[0]

                    l_path = self.labels_path.format(t_i)
                    p_path = self.preds_path.format(t_i)
                    self._write_labels(labels, num_total, l_path)
                    self._write_labels(higherprobclass, num_total, p_path)

                acc = 100.0 * num_dom / num_denom
                mlflow.log_metric(f'Test {t_i + 1} accuracy', acc.item())
                logger.info(f'Test accuracy for {num_total} data point: '
                            f'{acc:.2f} %')
                self.make_confusion_matrix(l_path, p_path, 'dataset {t_i}')

    def prediction(self):
        """Make prediction on prediction datasets"""

        self.graph.eval()
        logger.info('Making predictions...')
        with torch.no_grad():
            for predidx, predloader in enumerate(self.prediction_loaders):
                p_path = self.preds_path.format(predidx)
                total_num = 0
                for data in predloader:
                    inputs = data[TENSOR_INPUT_KEY_STR]
                    outputs = self.graph(inputs)

                    preds = torch.argmax(outputs, dim=1)

                    self._write_labels(preds, total_num, p_path)
                    total_num += pd.DataFrame(preds,
                                              columns=['predictions']).shape[0]

                logger.info(f'Predictions of {total_num} samples successfully '
                            f'saved to file {p_path}')

    def _load_weights(self):
        """Load network's weights if specified in parameters"""""

        if self.params['model']['weights']:
            w_path = cnst.WEIGHTS_PATH
            load_path = str(w_path / self.params['model']['weights'])
            if self.device_name == 'cpu':
                checkpoint = torch.load(load_path, map_location='cpu')
            else:
                checkpoint = torch.load(load_path)
            self.graph.load_state_dict(checkpoint['model_state_dict'])
            logger.debug('Weights loaded to the model: {0}'.format(load_path))

    def _set_optimiser(self):
        """Set up the optimiser based on parameters given"""

        opt_pars = self.params['training']['optimiser']
        if opt_pars['name'] == 'SGD':
            self.optimizer = SGD(self.graph.parameters(),
                                 lr=opt_pars['learning_rate'],
                                 momentum=opt_pars['momentum'],
                                 weight_decay=opt_pars['weight_decay'])
        elif opt_pars['name'] == 'adam':
            self.optimizer = optim.AdamW(self.graph.parameters(),
                                         lr=opt_pars['learning_rate'],
                                         betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=opt_pars['weight_decay'],
                                         amsgrad=False)
        elif opt_pars['name'] == 'radam':
            self.optimizer = RAdam(self.graph.parameters(),
                                   lr=opt_pars['learning_rate'],
                                   betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=opt_pars['weight_decay'])
        else:
            raise AttributeError(f"optimiser {opt_pars['name']} is not "
                                 f"supported.")

    def _set_scheduler(self):
        """Set up the scheduler based on parameters given"""

        opt_pars = self.params['training']['optimiser']
        if opt_pars['scheduler'] is None:
            self.scheduler = None
        elif opt_pars['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        elif opt_pars['scheduler'] == 'cycliclr':
            self.scheduler = optim.CyclicLR(self.optimizer,
                                            mode='exp_range',
                                            base_lr=opt_pars['base_lr'],
                                            max_lr=opt_pars['max_lr'])
        else:
            raise AttributeError(f"Scheduler {opt_pars['scheduler']} is not "
                                 f"supported.")

    def _set_criterion(self):
        """Set up the criterion based on parameters given"""

        self.criterion = nn.NLLLoss(self.class_weights.float()).to(self.device)

    def _write_labels(self, labels, base_index, save_path, map_it=True):
        """Write labels to a .csv file"""

        # overwrite the file if prediction from base 0
        mode = 'a' if base_index != 0 else 'w'

        if map_it:
            labels = list(map(self._map_preds, labels))
        labels = pd.DataFrame(labels, columns=['predictions'])
        labels.index += base_index
        with open(save_path, mode) as f:
            labels.to_csv(f, header=False)

    def _save_state(self, epoch):
        """Save the model and current state of training"""

        if not self.save_model or not self.is_best:
            return

        exp_path = cnst.EXPERIMENT_PATH
        save_path = str(exp_path / f'checkpoint_epoch{epoch}.pth.tar')

        torch.save({
            'epoch': epoch,
            'n_best_accs': self.n_best_accs,
            'class_weights': self.class_weights,
            'model_state_dict': self.graph.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        logger.debug('Model saved to {0}'.format(save_path))

    def _resume_state(self):
        """Resume state of the training if specified by parameters"""

        if not self.params['training']['resume']:
            return

        assert self.params['training']['checkpoint'], \
            'If resume, then checkpoint path must be provided'

        checkpoint = torch.load(str(self.resume_checkpoint_path))
        self.graph.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.class_weights = checkpoint['class_weights']
        self.n_best_accs = checkpoint['n_best_accs']
        self.is_best = True

        self.graph.to(self.device)
        res_path = self.resume_checkpoint_path
        logger.debug(f'Training state loaded to resume: {res_path}')

    def _get_class_weights(self, train_data):
        """Class weights used for classification in case of unbalanced data"""

        if 0 in train_data.class_count:
            logger.debug('Class count has a class with zero occurrences. '
                         'data_length/0.1 will be considered as class weight')

        class_count = np.array([count if count > 0 else 0.1
                                for count in train_data.class_count])
        class_weights = train_data.data_length / class_count
        class_weights = class_weights / np.sum(class_weights)
        class_weights = torch.from_numpy(class_weights)
        self.class_weights = class_weights

    @staticmethod
    def _set_loaders(datasets):
        loaders = []
        for dataset in datasets:
            loaders.append(dataset.get_loader())
        return loaders

    def make_confusion_matrix(self, labelpath, predspath, title):
        logger.debug(f'Making confusion matrices for {title}')
        labels = pd.read_csv(labelpath,
                             usecols=[0, 1],
                             names=['id', 'predictions'],
                             index_col='id')
        preds = pd.read_csv(predspath,
                            usecols=[0, 1],
                            names=['id', 'predictions'],
                            index_col='id')

        if not np.issubdtype(labels.dtypes['predictions'], np.integer):
            labels = labels.replace({"predictions": self._get_total_mapping()})
            preds = preds.replace({"predictions": self._get_total_mapping()})
            labels = np.array(labels['predictions'], dtype=np.int32)
            preds = np.array(preds['predictions'], dtype=np.int32)

        exp_path = cnst.EXPERIMENT_PATH
        conf_fig = CONFUSION_FIG_FILE + f"_{title.replace(' ','')}"
        normfpath = str(exp_path / (conf_fig + "_normed.png"))
        figpath = str(exp_path / (conf_fig + ".png"))

        classes = np.arange(self.num_classes)
        title = f'Confusion matrix, {title}'
        fig1 = plot_confusion_mat(labels, preds, classes, True, title=title)
        fig2 = plot_confusion_mat(labels, preds, classes, False, title=title)
        fig1.savefig(normfpath)
        fig2.savefig(figpath)

        logger.debug(f'Confusion matrices saved to {normfpath} and {figpath}')

    def __str__(self):
        if self.is_train:
            return f"""Model class {self.__class__.__name__}:
                   {str(self.params['model'])}
                   training with parameters:
                   {self.params['training']}
                   """
        else:
            return f"""Model class {self.__class__.__name__}:
                   {str(self.params['model'])}"""
