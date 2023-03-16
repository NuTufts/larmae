#!/bin/sh

# We assume we are already in the container
LARMAE_DIR=/n/home01/twongjirad/larmae
ICDL_DIR=/n/home01/twongjirad/icdl

cd ${ICDL_DIR}
source setenv_py3.sh
source configure.sh

cd $LARMAE_DIR
python3 train.py -c config_train.yaml -g 4


