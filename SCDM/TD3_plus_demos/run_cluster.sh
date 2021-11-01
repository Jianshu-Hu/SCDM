#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

#env_name="TwoEggCatchUnderArm-v0"
#env_name="EggCatchUnderarm-v0"
#env_name="EggCatchOverarm-v0"
#env_name="EggHandOver-v0"
#env_name="BlockCatchUnderarm-v0"
#env_name="BlockCatchOverarm-v0"
#env_name="BlockHandOver-v0"
env_name="PenHandOver-v0"

tag=PenHandOver-v0

FILE=output/$tag.txt
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    touch $FILE
    echo "$FILE was created."
fi


#CUDA_VISIBLE_DEVICES=0
#--add_invariance
#--use_normaliser
#--use_her
python main.py --env=$env_name --beta=0.7 --use_normaliser --expt_tag="$tag" --save_model | tee $FILE
