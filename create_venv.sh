#!/bin/bash
# this script is intended to be run where it is found

export DATA_DIR=/home/npetroce/data/
export PYTHONPATH=$PYTHONPATH:$PWD
python3.9 -m pip install virtualenv
python3.9 -m virtualenv models_env
source meta_env/bin/activate
pip install -U pip
pip install --default-timeout=1000 -r  requirements.txt

echo environment ready, activate with "source meta_env/bin/activate" next time you log in