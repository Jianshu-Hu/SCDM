#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

#env_name="Reacher-v2"
#env_name="Pusher-v2"
env_name="HalfCheetah-v3"

#env_name="PenSpin-v0"
#env_name="TwoEggCatchUnderArm-v0"

#env_name="EggCatchUnderarm-v0"
#env_name="EggCatchUnderarmHard-v0"
#env_name="EggCatchOverarm-v0"
#env_name="EggHandOver-v0"

#env_name="EggCatchUnderarm-v1"
#env_name="EggCatchOverarm-v1"

#env_name="EggCatchOverarm-v2"

#env_name="EggCatchOverarm-v3"
#env_name="EggCatchUnderarm-v3"

#env_name="BlockCatchUnderarm-v0"
#env_name="BlockCatchOverarm-v0"
#env_name="BlockHandOver-v0"

#env_name="PenHandOver-v0"
#env_name="PenCatchUnderarm-v0"
#env_name="PenCatchOverarm-v0"

#tag=random_goal_demo_5
tag=decaying_clipped_gaussian_noise_filter_with_max_diff_so_far
seed=3

demo_tag=""

#FILE=output/$tag.txt
#if [ -f "$FILE" ]; then
#    echo "$FILE exists."
#else
#
#    echo "$FILE does not exist."
#    touch $FIL3
#    echo "$FILE was created."
#fi


#CUDA_VISIBLE_DEVICES=0
#--use_normaliser
#--demo_goal_type="True
#--sparse_reward
#--initialize_with_demo
#--add_bc_loss

#--add_invariance_traj
#--add_invariance_regularization
#--add_hand_invariance_regularization
#--use_invariance_in_policy
#--add_artificial_transitions
#--enable_exploratory_policy
#--N_artificial_sample
#--inv_type="translation"
#--use_informative_segment
#--add_invariance --inv_type="translation" --use_informative_segment

echo "start running $env_name $tag with seed $seed"
#python main.py --add_artificial_transitions --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model

#reacher
#python main.py --eval_freq=100 --max_timesteps=100000 --model_start_timesteps=1000 --start_timesteps=5000 --without_demo --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model

#pusher
#python main.py --add_artificial_transitions --eval_freq=250 --max_timesteps=500000 --model_start_timesteps=5000 --start_timesteps=10000 --without_demo --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model

#cheetah
python main.py --add_artificial_transitions --eval_freq=200 --max_timesteps=200000 --model_start_timesteps=5000 --start_timesteps=10000 --without_demo --seed=$seed --use_normaliser --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --save_model

#| tee $FILE
