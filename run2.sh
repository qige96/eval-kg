#!/bin/bash

conda_env='ampligraph-v2'
expname='exp7'
export DATA_HOME=${PWD}/ampligraph_datasets
OUTPUT_HOME=${PWD}/expres/

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

source /home/${USER}/miniconda3/bin/activate $conda_env


# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_SCRATCH=${SCRATCH_HOME}/eval-kg/ampligraph_datasets

mkdir -p ${DATA_SCRATCH}
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

echo "Creating directory to save outputs"
export OUTPUT_DIR=${SCRATCH_HOME}/eval-kg/expres/${expname}
mkdir -p ${OUTPUT_DIR}


# ====================
# Run the job
# ====================
python __init__.py -i ${DATA_SCRATCH}  -o ${OUTPUT_DIR}

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}

echo "Job ${SLURM_JOB_ID} is done!"
