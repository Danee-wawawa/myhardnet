#!/bin/bash
cd ..
python ./code/HardNet.py --num_neg=3 --training-set=notredame --log-dir=../notredame_logs/ --model-dir=../notredame_models/ --experiment-name=/notredame_train/ --gpu-id=2 --fliprot=False | tee -a log_notredame_noaug.log
python ./code/HardNet.py --num_neg=3 --training-set=yosemite --log-dir=../yosemite_logs/ --model-dir=../yosemite_models/ --experiment-name=/yosemite_train/ --gpu-id=2 --fliprot=False | tee -a log_yosemite_noaug.log

