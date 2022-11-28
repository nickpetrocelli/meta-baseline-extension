#!/bin/bash
#
#  Execute from the current working directory
# -cwd
#
#  This is a long-running job
# -l inf
#
#
#
# notify
# -m abes
#
export PATH=/home/npetroce/Desktop/classwork/CS2952C/models-extension/models_env/bin:$PATH
export DATA_DIR=/home/npetroce/data/
source meta_env/bin/activate
python3 train_classifier.py --config configs/train_classifier_tiered.yaml --gpu 0,1 
