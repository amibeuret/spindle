from abc import ABC, abstractmethod
from shutil import copyfile
from os.path import isfile as pathisfile
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy as np2torch
import h5py
import yaml

import deepsleep.configs.constants as cnst
from deepsleep import ROOT_LOGGER_STR
from deepsleep import (TENSOR_INPUT_KEY_STR, TENSOR_OUTPUT_KEY_STR)
from deepsleep import (HDF5_KEY_INPUT_DST_STR, HDF5_KEY_OUTPUT_DST_STR,
                       HDF5_KEY_META_DST_STR)


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class BaseLoader(ABC, Dataset):
    """Load data from folders and create HDF5 files

    This class must be inherited to read data from folders, make hdf5 files
    out of it, and return the corresponding :class:`torch.DataLoader` to
    feed the network. This is the class given to the
    :class:`deepsleep.models.BaseModel` to feed the network. The main
    benefit of this class is to take care of creating and reading from HDF5
    files.

    In addition to the parameters below, paths attributes of
    :const:`deepsleep.configs.constants` must be set (see
    :func:`deepsleep.utils.set_up_paths()`)

    :param config: dictionary containing parameters required in this class

        ``"hdf5"``
            Name of the hdf5 to be saved
        ``"num_workers"``
            number of workers given to :class:`torch.DataLoader`
        ``"do_shuffle"``
            if True. then :class:`torch.DataLoader` will shuffle the data
        ``"batch_size"``
            batch size given to :class:`torch.DataLoader`
        ``"batch_prefetch"``
            Number of batches that will be pre-fetched to internal
            cache array from HDF5
        ``"folds_path"`` (optional)
            path to the .csv file containing information about each
            fold. Each column will have the name of hdf5 folds to be
            read at training each fold. Here "training fold" refers to
            the usual folds used in machine learning terminology,
            "hdf5 folds" refer to each dataset saved separately in the
            hdf5 file.
        ``"folds"``
            list of the folds to be feed to the network. Each hdf5 dataset
            is consisted of one or more folds, of which a subset can be
            feed to the network. ``"folds"`` can be a list of strings to
            specify each fold in hdf5, or if ``"folds_path"`` is provided,
            it should be a list of integers which correspond to the columns
            of the .csv file
    :type config: dict
    :param preprocessing: Instance of one of the preprocessing classes
        defined in :mod:`deepsleep.data.preproc` (must implement
        :func:`__call__(all_signals, srate, eeg_idx, eog_idx, emg_idx)` to
        preprocess data)
    :Example:

        >>> pred_handlers = ProdData(DATALOADER_PARAMS, preprocessing)
        >>> pred_handlers.make_hdf5()
        >>> data_loader = pred_handlers.get_loader()
    """

    def __init__(self, config, preprocessing):
        self.config = config
        self.preprocessing = preprocessing

        # Set by the child class
        self.hdf5_num_classes = None

        # DataLoader and loading parameters
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.do_shuffle = config['do_shuffle']

        self.batch_prefetch = config['batch_prefetch']

        # Path related variables
        file_name = preprocessing.name + '-' + config['hdf5']
        self.data_root = cnst.DATA_PATH.with_suffix('')
        self.partition_path = self.data_root / file_name
        self.temp_partition_path = cnst.TMP_DIR / file_name

        self.folds = self.set_folds_from_config(self.data_root, config)

        # These are set in 'get_loader()' before returning the DataLoader.
        # They correspond to the data specified for the relevant folds.
        self.data_length = None
        self.input_dim = None
        self.class_count = None
        self._fold_indices = None

        # Cache data must be set in '__item__(idx)'. This way they
        # correspond to the process executing them in case of multiple
        # workers. Otherwise there will be inconsistencies.
        self._cache_max_idx = -1
        self._base_cache_idx = 0
        self._cached_inputs = []
        self._cached_outputs = []
        self._cached_fold = ''

    @abstractmethod
    def read_all_data(self):
        """ Read data from data directory

        Each dataset have their own specific file hierarchy and thus reading
        the dataset need customised code to iterate on source files and load
        relevant EEG/EMG/EOG data. This method must be implemented for each
        different dataset to read data.
        The method should yield '(fold_name, data, labels)' where
        'fold_name' specifies the name of the database in hdf5 for the
        corresponding 'data' and 'labels'. 'data' is preprocessed data ready
        to be fed to the network.
        'fold_name' is used for later possible cross-validation. If this is
        not the case, then one default 'fold_name' will be enough.

        :return foldname: name of the cross-validation fold to which `data`
            and `labels` belong
        :type foldname: str
        :return data: Numpy array of preprocessed EEG/EMG/EOG data of shape
            (n, c, ...) where n is number of samples, c number of channels (
            from 1 to 3 specificying EEGs, EMGs and EOGs), and the rest number
            of features for each channel.
        :type data: ndarray
        :return labels: Numpy array of labels corresponding to the `data` of
            shape (n, ...) where n is the number of samples the same is in
            `data`
        :type labels: ndarray
        """

        pass

    @abstractmethod
    def transform_data(self, data, labels):
        """Data manipulations after being read from hdf5 file

        This is called after making the hdf5 while it's being read,
        to increase the flexibility of data manipulation even after saved 
        hdf5 data. If not required, you could simply use an identity 
        transformation: 'return data, labels'
        It is important to modify the metadata such as data length,
        class counts etc. to correspond to the new changes made by this
        function. This can be achieved through implementing
        :meth:`.transform_meta()` which is called in the :meth:`.get_loader(
        )`

        :param data: Numpy array of preprocessed EEG/EMG/EOG data of shape
            (n, c, ...) where n is number of samples, c number of channels (
            from 1 to 3 specificying EEGs, EMGs and EOGs), and the rest number
            of features for each channel.
        :type data: ndarray
        :param labels: Numpy array of labels corresponding to the `data` of
            shape (n, ...) where n is the number of samples the same is in
            `data`
        :type labels: ndarray
        :return data: same as the parameter `data`
        :return labels: same as the parameter `labels`
        """

        pass

    @abstractmethod
    def transform_meta(self):
        """ Make necessary changes to this object

        This is called once before returning the final
        :class:`torch.DataLoader`, to make any changes to the class if
        necessary. These changes may have been due to data manipulations in
        :meth:`.transform_data()`. If :meth:`.transform_data()` is the
        identity function, then an identity function here is enough.
        """
        pass

    def get_loader(self):
        """Return torch `DataLoader`

        It also sets the dataset related fields: `input_dim`, `data_length`,
        `class_count` and `fold_indices`. The latter is used to find the
        fold in which each element exists.

        If :meth:`.make_hdf5()` is already called, then HDF5 file exists and
        the data loader is initiated with the self object, where data is
        read from the HDF5 file. If the HDF5 files does not exist, then the
        data loader is initiated directly from the source files.
        """

        if not self._hdf5_exists_and_is_valid():
            dtensors = []
            ltensors = []
            for fold_name, data, labels in self.read_all_data():
                if fold_name not in self.folds:
                    continue

                dtensors.append(torch.from_numpy(data).float())
                ltensors.append(torch.from_numpy(labels).float())

            tensordata = torch.cat(dtensors, dim=0)
            tensorlabels = torch.cat(ltensors, dim=0)
            self.input_dim = tensordata.shape[1:]
            self.data_length = tensordata.shape[0]
            self.class_count = None
            dataset = CustomTensorDataset((tensordata, tensorlabels))
            return DataLoader(dataset,
                              batch_size=self.batch_size,
                              shuffle=self.do_shuffle,
                              num_workers=self.num_workers)

        hdf5file = str(self.partition_path)
        with h5py.File(hdf5file, 'r') as hf:
            total_meta = yaml.load(hf[HDF5_KEY_META_DST_STR][()],
                                   Loader=yaml.FullLoader)
            num_classes = len(total_meta['class-count'])
            self.input_dim = total_meta['input-dim']
            self.data_length = 0
            self.class_count = np.zeros(num_classes)
            fold_lengths = []
            new_folds = []
            for fold in self.folds:
                meta_key_str = fold + '/' + HDF5_KEY_META_DST_STR
                if fold not in hf.keys():
                    logger.warning(f"fold {fold} not found in dataset. "
                                   f"Skipping this fold")
                    continue
                fold_meta = yaml.load(hf[meta_key_str][()],
                                      Loader=yaml.FullLoader)
                fold_lengths.append(fold_meta['data-length'])
                self.data_length += fold_meta['data-length']
                self.class_count += fold_meta['class-count']
                new_folds.append(fold)

        self._fold_indices = np.cumsum(fold_lengths)
        self.folds = new_folds  # In case some folds did not exist in hdf5
        self.transform_meta()

        if (cnst.TMP_DIR.is_absolute()
                and not self.temp_partition_path.is_file()):
            logger.info(f"Copy file "
                        f"from {self.partition_path} "
                        f"to {self.temp_partition_path}...")
            copyfile(str(self.partition_path), str(self.temp_partition_path))
        return DataLoader(self, batch_size=self.batch_size,
                          shuffle=self.do_shuffle,
                          num_workers=self.num_workers)

    def __getitem__(self, idx):
        """Return the i'th (idx) element of the entire dataset.

        It first finds out in which fold the idx exists, then loads a
        specified chunk of that fold and keeps it as cached array. Finally it
        returns the i'th element from this cached array.
        The chunk is in range [idx, idx + self.batch_prefetch]
        """

        if self._cache_max_idx <= idx or idx < self._base_cache_idx:
            self._update_cache_and_transform(idx)

        cache_idx = idx - self._base_cache_idx
        input_sample = self._cached_inputs[cache_idx]
        output_sample = []
        if len(self._cached_outputs) > 0:
            output_sample = np.array(self._cached_outputs[cache_idx])
        ret_val = {TENSOR_INPUT_KEY_STR: np2torch(input_sample).float(),
                   TENSOR_OUTPUT_KEY_STR: np2torch(output_sample).long()}

        return ret_val

    def __len__(self):
        return self.data_length

    def _update_cache_and_transform(self, idx):
        """ Update the cached array

        If global idx is not in the range of the cached data, update the
        cache from idx to (idx + batch_prefetch * batch_size) or the end of
        the fold data.
        """

        if self._cache_max_idx > idx >= self._base_cache_idx:
            return

        hdf5file = str(self.temp_partition_path if cnst.TMP_DIR.is_absolute()
                       else self.partition_path)

        # Find the fold db which has this idx
        fold_index = np.where((idx < self._fold_indices))[0][0]

        # Previous is used to find the relative idx in this fold
        previous_index = fold_index - 1
        if previous_index < 0:
            previous_index = 0
        else:
            previous_index = self._fold_indices[previous_index]

        fold = str(self.folds[fold_index])
        with h5py.File(hdf5file, 'r') as hf:
            input_ds = hf.get(fold + '/' + HDF5_KEY_INPUT_DST_STR)
            output_ds = hf.get(fold + '/' + HDF5_KEY_OUTPUT_DST_STR)

            # Get the relative indices for this specific cache
            base_idx = idx - previous_index
            max_idx = min(base_idx + self.batch_prefetch * self.batch_size,
                          input_ds.shape[0])

            # Update all cache variables
            self._base_cache_idx = idx
            self._cache_max_idx = previous_index + max_idx
            self._cached_fold = fold
            inputs = input_ds[base_idx: max_idx]
            outputs = []
            if len(output_ds) > 0:
                outputs = output_ds[base_idx: max_idx]

        self._cached_inputs, self._cached_outputs = self.transform_data(
            inputs, outputs)

    def make_hdf5(self):
        """ Make hdf5 file out of the dataset

        Iterate on data folders and save preprocessed data into hdf5
        file. The iterator returns 'fold_name' for each pair of data and label
        which then determines the corresponding database in the hdf5 file.
        This is used later for cross-validation
        """
        
        if self._hdf5_exists_and_is_valid():
            return

        logger.debug(f'Saving preprocessed data into {self.partition_path}...')
        
        fold_class_count = np.zeros(self.hdf5_num_classes)
        total_class_count = np.zeros(self.hdf5_num_classes)
        total_data_length = 0
        fold_data_length = 0
        old_fold_name = ''
        data_shape = ()
        for fold_name, data, labels in self.read_all_data():
            logger.debug(f'Saving fold {fold_name}...')
            if old_fold_name and old_fold_name != fold_name:
                # Close the previous fold
                self._hdf5_fold_metadata(old_fold_name,
                                         fold_class_count,
                                         fold_data_length)

                total_class_count += fold_class_count
                total_data_length += fold_data_length

                fold_data_length = 0
                fold_class_count = np.zeros(self.hdf5_num_classes)
            old_fold_name = fold_name

            self._create_or_write_hdf5(fold_name, data, labels)
            fold_data_length += labels.shape[0]
            fold_class_count += np.histogram(
                labels, bins=self.hdf5_num_classes)[0]
            data_shape = data.shape[1:]

        # For last iteration
        self._hdf5_fold_metadata(
            old_fold_name, fold_class_count, fold_data_length)
        total_class_count += fold_class_count
        total_data_length += fold_data_length
        self._hdf5_total_metadata(
            total_class_count, total_data_length, data_shape)

    def _hdf5_exists_and_is_valid(self):
        """ Check if hdf5 exists and is valid

        :return: 'True' if the hdf5 file already exists and has consistent
        parameters. Otherwise 'False'
        :rtype: bool
        """

        if not self.partition_path.is_file():
            return False

        param_error = ('Currently there exists an hdf5 file on disk with the '
                       'same name but different parameters. This old file '
                       'will be replaced with a new file and updated '
                       f'parameters \nparameters: {self.partition_path}')

        broken_error = (f'dataset {self.partition_path} seemed to  exist but '
                        f'broken. It is removed and a new one will be '
                        f'created.')
        # Check if the file on disk is consistent with current parameters
        with h5py.File(str(self.partition_path), 'r') as hf:
            if HDF5_KEY_META_DST_STR not in hf:
                self.partition_path.unlink()
                logger.warning(broken_error)
                return False

            metadata = yaml.load(hf[HDF5_KEY_META_DST_STR][()],
                                 Loader=yaml.FullLoader)
            if not metadata['saved-successfully']:
                self.partition_path.unlink()
                logger.warning(broken_error)
                return False

            folds = [fold for fold in hf.keys() if fold.startswith('fold')]
            for fold in folds:
                meta_key = fold + '/' + HDF5_KEY_META_DST_STR
                input_dataset = hf.get(fold + '/' + HDF5_KEY_INPUT_DST_STR)
                metadata = yaml.load(hf[meta_key][()], Loader=yaml.FullLoader)

                if input_dataset.shape[0] != metadata['data-length']:
                    self.partition_path.unlink()
                    logger.warning(param_error)
                    return False

                if metadata['preprocessing'] != str(self.preprocessing):
                    self.partition_path.unlink()
                    logger.warning(param_error)
                    return False

                if ('class-count' in metadata.keys()
                        and input_dataset.shape[0]
                        != np.sum(metadata['class-count'])):

                    self.partition_path.unlink()
                    logger.warning(param_error)
                    return False

        logger.debug((f'dataset {self.partition_path} is already prepared '
                      f'and stored on disk.'))
        return True

    def _create_or_write_hdf5(self, fold_name, data, labels):
        """Append or create dataset corresponding to a fold group"""

        in_str = fold_name + '/' + HDF5_KEY_INPUT_DST_STR
        o_str = fold_name + '/' + HDF5_KEY_OUTPUT_DST_STR
        if not self.partition_path.is_file():  # Create the hdf5
            with h5py.File(str(self.partition_path), 'w') as hf:
                sh = data.shape
                hf.create_dataset(in_str,
                                  data=data,
                                  dtype='f8',
                                  maxshape=(None, sh[1], sh[2], sh[3]))
                hf.create_dataset(o_str,
                                  data=labels,
                                  dtype='u1',
                                  maxshape=(None,))

        else:  # hdf5 file already exists. Append to it
            with h5py.File(str(self.partition_path), 'a') as hf:
                if in_str in hf:
                    hf[in_str].resize((hf[in_str].shape[0] + data.shape[0]),
                                      axis=0)
                    hf[in_str][-data.shape[0]:, ...] = data
                    hf[o_str].resize((hf[o_str].shape[0] + labels.shape[0]),
                                     axis=0)
                    hf[o_str][-labels.shape[0]:] = labels
                else:
                    sh = data.shape
                    hf.create_dataset(in_str,
                                      data=data,
                                      dtype='f8',
                                      maxshape=(None, sh[1], sh[2], sh[3]))
                    hf.create_dataset(o_str,
                                      data=labels,
                                      dtype='u1',
                                      maxshape=(None,))

    def _hdf5_fold_metadata(self, fold_name, class_count, data_length):
        """ Write metadata related to the specific `fold`

        This is the metadata related to a specific fold group.
        'fold_name' specifies the name of the fold dataset, 'class_count'
        and 'data_length' must correspond to the specific fold

        :param fold_name: name of this fold
        :type fold_name: str
        :param class_count: count of each class for this fold. A numpy array
            of shape (c) where c is the number of existing classes
        :type class_count: ndarray
        :param data_length: length of the data in this fold
        :type: int
        """

        meta_key_str = fold_name + '/' + HDF5_KEY_META_DST_STR
        input_key_str = fold_name + '/' + HDF5_KEY_INPUT_DST_STR
        output_key_str = fold_name + '/' + HDF5_KEY_OUTPUT_DST_STR

        metadata = dict()
        metadata['class-count'] = class_count.tolist()
        metadata['preprocessing'] = str(self.preprocessing)
        metadata['data-length'] = data_length
        metadata['saved-successfully'] = True

        with h5py.File(str(self.partition_path), 'a') as hf:
            hf.create_dataset(meta_key_str, data=yaml.dump(metadata))
            hf[input_key_str].resize((hf[input_key_str].shape[0]), axis=0)
            hf[output_key_str].resize((hf[output_key_str].shape[0]), axis=0)

        logger.debug(f'Fold {fold_name} of HDF5 file successfully created:'
                     f'{self.partition_path}')

    def _hdf5_total_metadata(self, class_count, data_length, data_shape):
        """ write metadata of the entire dataset to hf5

        This is the aggregate metadata related to all folds.
        'class_count' and 'data_length' correspond to the entire dataset. (
        Sum of folds together). 'data_shape' is the shape of the input data
        ignoring the batch_size

        :param class_count: count of each class for this fold. A numpy array
            of shape (c) where c is the number of existing classes
        :type class_count: ndarray
        :param data_length: length of the data in this fold
        :type: int
        :param data_shape: shape of the input data ignoring the sample
            number. This will be used to initialise the nn.torch model.
        :type data_shape: tuple
        """

        metadata = dict()
        metadata['class-count'] = class_count.tolist()
        metadata['preprocessing'] = str(self.preprocessing)
        metadata['data-length'] = data_length
        metadata['input-dim'] = data_shape
        metadata['saved-successfully'] = True

        with h5py.File(str(self.partition_path), 'a') as hf:
            hf.create_dataset(HDF5_KEY_META_DST_STR, data=yaml.dump(metadata))

        logger.debug(f'HDF5 file successfully created: {self.partition_path}')

    @staticmethod
    def set_folds_from_config(data_root, configs):
        """ Parse folds from the config file

        If `folds_path` exists in configs, then get folds from the csv
        file, otherwise folds are the same as 'configs['folds']'. In the
        case of `folds_path`, content of `folds` must be integers indicating
        column numbers of the csv file
        """

        if 'folds_path' in configs and configs['folds_path']:
            assert type(configs['folds_path']) == str
            assert all(isinstance(x, int) for x in configs['folds'])

            csv_p = (configs['folds_path'] if pathisfile(configs['folds_path'])
                     else data_root / configs['folds_path'])
            assert pathisfile(csv_p)

            csv_f = pd.read_csv(csv_p, header=None)
            folds = [csv_f[idx].dropna().tolist() for idx in configs['folds']]
            folds = list(np.concatenate(folds).astype('str'))
            logger.debug(f"folds are converted from "
                         f"{configs['folds_path']} to {folds}")
        else:
            folds = configs['folds']

        return folds


class CustomTensorDataset(Dataset):
    """TensorDataset to return key values

    This is implemented to make on-the-fly feeding compatible with feeding
    from hdf5 files
    """

    def __init__(self, tensors):

        self.tensors = tensors

    def __getitem__(self, index):
        tensors = self.tensors
        input_sample = tensors[0][index]

        output_sample = np2torch(np.array([]))
        if len(tensors[1]) > 0:
            assert all(tensors[0].size(0) == tensor.size(0)
                       for tensor in tensors)
            output_sample = tensors[1][index]

        ret_val = {TENSOR_INPUT_KEY_STR: input_sample.float(),
                   TENSOR_OUTPUT_KEY_STR: output_sample.long()}
        return ret_val

    def __len__(self):
        return self.tensors[0].size(0)
