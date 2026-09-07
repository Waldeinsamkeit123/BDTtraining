#!/bin/bash

#SBATCH --job-name=ttH-0L-BDT-v4
#SBATCH --partition=gpu
#SBATCH --account=cmsgpu
#SBATCH --qos=cmsnormal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-ttH-0L-BDT-v4-%j.out

set -euo pipefail

PROJECT=/scratchfs2/cms/wanghan/BDTtraining
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT}"

python scripts/train_bdt.py \
    --config config/bdt_0L_v4_sameinfo.json \
    --device cpu

