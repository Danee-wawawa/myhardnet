#!/bin/bash
cd ..
python ./code/HardNet.py --fliprot=False --experiment-name=/liberty_train/ | tee -a log_HardNet_Lib.log
python ./code/HardNet.py --fliprot=True --experiment-name=/liberty_train_with_aug/  | tee -a log_HardNetPlus_Lib.log
