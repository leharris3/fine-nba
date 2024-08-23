#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=mirage.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=50:00:00

# PROJ_ROOT=/mnt/opr/levlevi/opr/fine-nba/src/statvu_align
VENV_NAME="align"
CONDA_PATH=/mnt/opr/levlevi/anaconda3/etc/profile.d/conda.sh
PYTHON_PATH="/mnt/opr/levlevi/anaconda3/envs/align/bin/python"
SCRIPT_PATH="pipeline_testing.py"

# cd $PROJ_ROOT
source $CONDA_PATH
conda activate $VENV_NAME

for rank in {0..7}
do
    $PYTHON_PATH $SCRIPT_PATH --rank $rank
done