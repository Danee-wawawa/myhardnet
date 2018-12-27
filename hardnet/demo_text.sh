#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=5
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#export PYTHONUNBUFFERED=1
#export MXNET_ENABLE_GPU_P2P=0
#export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

MODEL_PATH=model
PREFIX=${MODEL_PATH}/final

python2 -m rcnn.tools.demo_images \
    --network insightext \
    --prefix ${PREFIX} \
    --epoch 0 \
    --gpu 1 \
    --thresh 0.3 \
    --vis


