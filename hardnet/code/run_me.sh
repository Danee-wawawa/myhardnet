#!/bin/bash
cd ..
python ./code/HardNet.py --training-set=notredame --gpu-id=2 --lr=10.0 --fliprot=False --experiment-name=/notredame_train/ | tee -a log_HardNet_Lib_notredame.log
