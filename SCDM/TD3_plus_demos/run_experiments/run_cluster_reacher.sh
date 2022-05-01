#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

env_name="Reacher-v2"

tag=MVE_H_1
seed=3

demo_tag=""

#CUDA_VISIBLE_DEVICES=0

echo "start running $env_name $tag with seed $seed"

#reacher
python main.py --add_artificial_transitions_type='MVE' --prediction_horizon=1 --eval_freq=100 --max_timesteps=100000 --model_start_timesteps=1000 --start_timesteps=5000 --without_demo --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model
