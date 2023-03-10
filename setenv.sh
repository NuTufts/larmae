#!/bin/sh

LARMAE_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${LARMAE_HOME}

cd /home/twongjirad/working/larbys/icarus/icdl
source setenv_py3.sh
source configure.sh

echo ${LARMAE_HOME}
cd ${LARMAE_HOME}
