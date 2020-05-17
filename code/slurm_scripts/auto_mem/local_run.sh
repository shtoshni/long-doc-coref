#!/bin/bash

venv=coref
base_dir=/home/shtoshni/Research/litbank_coref/code
sub_dir=auto_memory_model
args="$@"

echo $args
source /home/shtoshni/Development/anaconda3/etc/profile.d/conda.sh
export TORCH_HOME=/home/shtoshni/.cache/torch
export PYTHONPATH=${base_dir%/}/:$PYTHONPATH

gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)"
echo $venv
conda activate ${venv}
# if [ "$gpu_name" == "GeForce GTX TITAN X" ];
# then
#     conda activate ${venv}
#     echo "Using environment ${venv}."
# else
#     conda activate "${venv}_10"
#     echo "Using environment ${venv}_10."
# fi


echo "Host: $(hostname)"
echo "GPU: $gpu_name"
echo "PYTHONPATH: $PYTHONPATH"
echo "--------------------"

echo "Starting experiment."
if [ -z "$SLURM_JOB_NAME" ];
then
    python ${base_dir%/}/${sub_dir%/}/main.py ${args}
else
    # NOT USING THE SLURM ID RIGHT NOW
    python ${base_dir%/}/${sub_dir%/}/main.py ${args} -slurm_id ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
fi

conda deactivate
