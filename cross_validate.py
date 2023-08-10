import re
import time
import argparse
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from deepsleep import EXPERIMENTS_FOLDER


def _extract_folds_from_csv(csv_path, fold_num=None):
    data_files = []
    if csv_path is None:
        return data_files

    csv_file = pd.read_csv(str(csv_path), header=None)
    if fold_num is None:
        num_folds = csv_file.shape[1]
        for idx in range(num_folds):
            data_files.append(csv_file[idx].dropna().tolist())
    else:
        data_files = csv_file[fold_num].dropna().tolist()
    return data_files


def _extract_test_results(file_path):
    accs = []
    fs = []
    with file_path.open(mode='r') as log_text:
        line = log_text.readline()
        while line:
            test_data = re.match(r".*start testing.* folds (.*)", line)
            acc_text = re.match(r".*Test accuracy for.* (\d*\.\d+)", line)
            if acc_text:
                accs.append(float(acc_text[1]))
            if test_data:
                fs.append(test_data[1])
            line = log_text.readline()
    if len(accs) != len(fs):
        print(f'{file_path} had inconsistent logs. Skipped entirely')
        return [], []

    return accs, fs


def visualise(args):
    fold_dirs = (args.result_dir / EXPERIMENTS_FOLDER).iterdir()
    fold_dirs = [fdir for fdir in fold_dirs
                 if fdir != 'configs' and fdir != 'mlruns']
    logs = [args.result_dir / EXPERIMENTS_FOLDER / fdir / 'logs.txt'
            for fdir in fold_dirs]
    logs = [log for log in logs if log.is_file()]
    accuracies = []
    folds = []
    for log in logs:
        [accs, fs] = _extract_test_results(log)
        accuracies.append(accs)
        folds.append(fs)

    folds = np.concatenate(folds)
    accuracies = np.concatenate(accuracies)
    df = pd.DataFrame({'folds': folds, 'accuracies': accuracies})
    out_csv = args.result_dir / 'cv_results.csv'
    df.to_csv(str(out_csv), sep='\t')


def cross_validate(args, configs):
    from_file = args.which == 'filepath'
    exp_id = 'cross_val' + '_' + str(time.time())
    exp_path = args.result_dir / EXPERIMENTS_FOLDER / exp_id
    configs_dir = exp_path / 'configs'
    train_path = ''
    test_path = ''
    configs_dir.mkdir(parents=True, exist_ok=True)
    if from_file:
        train_path = (args.train_path if args.train_path.is_file()
                      else args.data_dir / args.train_path)
        test_path = (args.test_path if args.test_path.is_file()
                     else args.data_dir / args.test_path)

    if from_file:
        train_folds = _extract_folds_from_csv(train_path)
    else:
        train_folds = list(permutations(args.folds, args.train_fold_nb))

    configs['training']['save_model'] = False
    train_config = configs['data']['train'].copy()
    test_config = configs['data']['test'].copy()
    for idx, train_fold in enumerate(train_folds):
        train_fold = list(train_fold)
        for rep in range(args.rep):
            # Get the test folds
            if from_file:
                test_folds = _extract_folds_from_csv(test_path, idx)
            else:
                test_folds = [fold for fold in args.folds
                              if fold not in train_fold]

            train_config['datasets'][0]['folds'] = train_fold
            train_config['datasets'][0]['folds_path'] = None
            train_config['datasets'][0]['merge_folds'] = True
            test_config['datasets'][0]['folds'] = test_folds
            test_config['datasets'][0]['folds_path'] = None
            test_config['datasets'][0]['merge_folds'] = \
                True if args.test_on_all else False

            configs['data']['train'] = train_config
            configs['data']['test'] = test_config

            # Save the new config file
            config_path = configs_dir / f'config_fold{idx}.yaml'
            with config_path.open(mode='w') as yaml_file:
                yaml.dump(configs, yaml_file)

            # Run training with the new config file
            command = (f"bsub -n 2 -W 12:00 "
                       f"-R rusage[mem=10000,scratch=80000,ngpus_excl_p=1] "
                       f"python run.py {exp_path} {config_path} "
                       f"--data_dir {args.data_dir} "
                       f"--tmpdir")
            subprocess.check_output(command.split())
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='types of cross-validation input')
    file_p = subparsers.add_parser("filepath")
    arg_p = subparsers.add_parser("params")
    vis_p = subparsers.add_parser("visualise")

    vis_p.set_defaults(which='vis')
    parser.add_argument("result_dir",
                        type=Path,
                        help="Path to the data and artefacts folder")
    parser.add_argument("--config",
                        type=Path,
                        help="Path to the main yaml config.")
    parser.add_argument("--data_dir",
                        type=Path,
                        help="Path to the data folder.")
    parser.add_argument('--test_on_all', action='store_true',
                        help="If specified, all test data folds will be "
                             "tested together instead of separately")
    parser.add_argument("--rep",
                        type=int,
                        default=1,
                        help="Number of repetitions on each fold")
    arg_p.add_argument('--folds', nargs='+', type=str,
                       help="list of all existing folds")
    arg_p.add_argument('--train_fold_nb', type=int, default=1,
                       help="Number of data folds to take for training "
                            "at each fold of cross validation")
    arg_p.set_defaults(which='params')
    file_p.add_argument('--train_path', type=Path,
                        help="Path to .csv file containing list of all "
                             "training folds. Each column is defined as"
                             " a separate fold where rows are the name "
                             "of dataset folds")
    file_p.add_argument('--test_path', type=Path,
                        help="Path to .csv file containing list of all "
                             "test folds. Each column is defined as "
                             "a separate fold where rows are the name "
                             "of dataset folds")
    file_p.add_argument('--valid_path', type=Path,
                        help="Path to .csv file containing list of all "
                             "validation folds. Each column is defined "
                             "as a separate fold where rows are the name"
                             " of dataset folds")
    file_p.set_defaults(which='filepath')
    args = parser.parse_args()

    if args.which == 'vis':
        visualise(args)
    else:
        with args.config.open(mode='r') as yaml_file:
            base_config = yaml.safe_load(yaml_file)

        cross_validate(args, base_config)


if __name__ == "__main__":
    main()
