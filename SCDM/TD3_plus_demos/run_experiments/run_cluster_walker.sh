#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

env_name="Walker2d-v3"
tag=SAC_auto_ours
seed=3

demo_tag=""

#CUDA_VISIBLE_DEVICES=0

echo "start running $env_name $tag with seed $seed"

#walker
python main.py --policy='SAC' --without_demo --add_artificial_transitions_type="ours" --prediction_horizon=3 --eval_freq=5000 --max_timesteps=3000000 --model_start_timesteps=5000 --start_timesteps=25000 --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model
