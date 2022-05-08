#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

env_name="PenSpin-v0"

#env_name="EggCatchUnderarm-v0"
#env_name="EggCatchOverarm-v0"

tag=DDPG
seed=3

demo_tag=""

#CUDA_VISIBLE_DEVICES=0

echo "start running $env_name $tag with seed $seed"
python main.py --policy='DDPG' --add_artificial_transitions_type='None' --prediction_horizon=1 --start_timesteps=50000 --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model
