#!/bin/bash

cd /bigdata/users/jhu/workspace/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

tag="with_symmetry_traj_one_thread"
FILE=output/with_symmetry_traj_one_thread_output.txt
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    touch $FILE
    echo "$FILE was created."
fi


#CUDA_VISIBLE_DEVICES=0
#--add_symmetry
python main.py --env="TwoEggCatchUnderArm-v0" --expt_tag=$tag --use_normaliser --save_model | tee $FILE
