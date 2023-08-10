module load python_gpu/3.7.1 hdf5/1.10.1
python cross_validate.py --config deepsleep/configs/train_caro_linus_config.yaml --data_dir $DATADIR/deepsleep/dataset params --folds fold0 fold1 fold2 fold3 fold4 $DATADIR/deepsleep/
python cross_validate.py --config deepsleep/configs/train_caro_linus_config.yaml --data_dir $DATADIR/deepsleep/dataset filepath --train_path $DATADIR/deepsleep/dataset/cv_train.csv --test_path $DATADIR/deepsleep/dataset/cv_test.csv $DATADIR/deepsleep/
python cross_validate.py visualise $DATADIR/deepsleep/experiments/
python gridsearch_leonhard.py $DATADIR/deepsleep/ deepsleep/configs/train_spindle_ss.yaml --data_dir $DATADIR/deepsleep/dataset --name SGD --learning_rate 0.1 0.01 0.001 0.0001
bsub -n 2 -W 12:00 -R rusage[mem=15000,scratch=80000,ngpus_excl_p=1] python run.py $DATADIR/deepsleep deepsleep/configs/train_spindle_ss.yaml --tmpdir
