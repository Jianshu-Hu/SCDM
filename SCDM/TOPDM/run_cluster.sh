#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TOPDM/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

env_name="EggCatchUnderarm-v1"
#env_name="BlockCatchOverarm-v0"
#env_name="BlockCatchUnderarm-v0"
#env_name="PenCatchOverarm-v0"
#env_name="PenCatchUnderarm-v0"
#env_name="PenHandOver-v0"
#env_name="TwoBlockCatchUnderArm-v0"
#env_name="TwoPenCatchUnderArm-v0"

tag=traj

#FILE=output/$tag.txt
#if [ -f "$FILE" ]; then
#    echo "$FILE exists."
#else
#    echo "$FILE does not exist."
#    touch $FILE
#    echo "$FILE was created."
#fi


#CUDA_VISIBLE_DEVICES=0
python trajectory_generator.py --expt_tag "$tag" --env $env_name --num_envs 10 --traj_len 75 --num_traj_generate 30
