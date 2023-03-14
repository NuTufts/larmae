#!/bin/sh

LARMAE_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${LARMAE_HOME}

MACHINE=`uname --nodename`

if [ $MACHINE == "pop-os" ]
then

    cd /home/twongjirad/working/larbys/icarus/icdl
    source setenv_py3.sh
    source configure.sh

else
    echo "DEFAULT SETUP (COMPAT WITH SINGULARITY CONTAINER)"    
fi
    
echo ${LARMAE_HOME}
cd ${LARMAE_HOME}

# ADD VIT-PYTORCH FOLDER INTO PYTHON PATH
[[ ":$PYTHONPATH:" != *":${LARMAE_HOME}/vit-pytorch:"* ]] && export PYTHONPATH="${LARMAE_HOME}/vit-pytorch:${PYTHONPATH}"
