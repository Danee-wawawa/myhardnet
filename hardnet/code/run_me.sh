#!/bin/bash
#cd ..
python HardNet.py --training-set=notredame --gpu-id=0 --lr=10.0 --fliprot=False --experiment-name=../../notredame_train/ | tee -a ../logs/log_HardNet_Lib_notredame.log
