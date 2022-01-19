#!/bin/bash

cd /bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/
source /bigdata/users/jhu/anaconda3/bin/activate
conda activate SCDM

#env_name="TwoEggCatchUnderArm-v0"

env_name="EggCatchUnderarm-v0"
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
tag=random_goal_demo_demo_divided_into_two_part_add_auto_regularization
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
#--N_artificial_sample
#--inv_type="translation"
#--use_informative_segment
#--add_invariance --inv_type="translation" --use_informative_segment

#--use_her
#--her_timesteps=1000000
#--N_her=1
#--her_type=1
#--use_her --her_timesteps=0 --her_type=2 --N_her=4
echo "start running $env_name $tag with seed $seed"
python main.py --seed=$seed --env=$env_name --expt_tag="$tag" --demo_tag=$demo_tag --demo_goal_type="Random" --add_hand_invariance_regularization --save_model

#| tee $FILE
