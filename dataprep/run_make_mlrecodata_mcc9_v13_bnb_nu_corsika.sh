#!/bin/bash

PREPDIR=/cluster/tufts/wongjiradlabnu/twongj01/larmae/dataprep/
WORKDIR=${PREPDIR}/workdir
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl
INPUTLIST=${PREPDIR}/mcc9_v29e_dl_run3_G1_extbnb_dlana_goodlist.txt
OUTPUT_DIR=${PREPDIR}/data/mcc9_v29e_dl_run3_G1_extbnb_dlana

TAG=run3extbnb


#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=40
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/${TAG}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/mlreco_dataprep_${TAG}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir
cd $local_jobdir
touch log_${TAG}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${TAG}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd $local_jobdir

SCRIPT="python3 ${PREPDIR}/make_training_images.py"
echo "SCRIPT: ${SCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..40}
do
    let lineno=$startline+$i
    extbnbfile=`sed -n ${lineno}p $INPUTLIST | awk '{print $1}'`
    COMMAND="${SCRIPT} -o out_${i}.root -i $extbnbfile"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
done

ls -lh out_*.root >> ${local_logfile} 2>&1

ana_outputfile=`printf "larmaedata_${TAG}_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root >> ${local_logfile} 2>&1
cp $ana_outputfile ${OUTPUT_DIR}/
cp ${local_logfile} ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
