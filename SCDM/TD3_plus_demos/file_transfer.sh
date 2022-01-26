source=/bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/saved_fig/
target=~/PycharmProjects/SCDM/SCDM/TD3_plus_demos/saved_fig/

#source=/bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/results_critic/
#target=~/PycharmProjects/SCDM/SCDM/TD3_plus_demos/results_critic/

source2=/bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/output/
target2=~/PycharmProjects/SCDM/SCDM/TD3_plus_demos/output/

file_name=*.png
file_name2=*
#file_name1=random_goal_demo_demo_divided_into_two_part_add_auto_regularization.npy

scp jhu@aaal.ji.sjtu.edu.cn:$source$file_name $target
scp jhu@aaal.ji.sjtu.edu.cn:$source2$file_name2 $target2

#scp $target jhu@aaal.ji.sjtu.edu.cn:/bigdata/users/jhu/SCDM/SCDM/TD3_plus_demos/new_plot.py
