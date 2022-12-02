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
export PATH=/home/npetroce/Desktop/classwork/CS2952Q/meta-baseline-extension/meta_env/bin:$PATH
export DATA_DIR=/home/npetroce/data/
source meta_env/bin/activate
python3 test_few_shot.py --config $1 --shot $2 --epsilon 0.2
