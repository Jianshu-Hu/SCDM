import numpy as np
import matplotlib.pyplot as plt
import math
overarm_prefix = "results/TD3_EggCatchOverarm-v0_"
overarm_critic_prefix = "results_critic/TD3_EggCatchOverarm-v0_"
underarm_prefix = "results/TD3_EggCatchUnderarm-v0_"
underarm_prefix_v1 = "results/TD3_EggCatchUnderarm-v1_"
underarm_critic_prefix_v1 = "results_critic/TD3_EggCatchUnderarm-v1_"
# overarm and underarm
tag_1 = ["0_random_goal_demo_1", "0_random_goal_demo_2", "0_random_goal_demo_3", "0_random_goal_demo_4", "0_random_goal_demo_5"]
tag_2 = ["random_goal_demo_with_normalizer_1", "random_goal_demo_with_normalizer_2", "random_goal_demo_with_normalizer_3"]

tag_7 = ["1_random_goal_demo_pr_init_0", "2_random_goal_demo_pr_init_0", "3_random_goal_demo_pr_init_0"]
tag_8 = ["1_random_goal_demo_pr_init_03", "2_random_goal_demo_pr_init_03", "3_random_goal_demo_pr_init_03"]

tag_3 = ["1_random_goal_demo_her_type_1_goal_from_demo", "2_random_goal_demo_her_type_1_goal_from_demo",
          "3_random_goal_demo_her_type_1_goal_from_demo"]
tag_4 = ["1_random_goal_demo_her_type_2_goal_from_demo", "2_random_goal_demo_her_type_2_goal_from_demo",
          "3_random_goal_demo_her_type_2_goal_from_demo"]

tag_5 = ["0_1_random_goal_demo_with_translation_regularization", "0_2_random_goal_demo_with_translation_regularization",
          "0_3_random_goal_demo_with_translation_regularization"]
tag_6 = ["0_1_random_goal_demo_with_rotation_regularization", "0_2_random_goal_demo_with_rotation_regularization",
          "0_3_random_goal_demo_with_rotation_regularization"]

tag_9 = ["1_random_goal_demo_her_type_1_goal_from_demo_successful_demos",
         "2_random_goal_demo_her_type_1_goal_from_demo_successful_demos",
         "3_random_goal_demo_her_type_1_goal_from_demo_successful_demos"]

tag_10 = ["0_1_random_goal_demo_replaced_by_001_translation", "0_2_random_goal_demo_replaced_by_001_translation",
         "0_3_random_goal_demo_replaced_by_001_translation"]
tag_11 = ["1_random_goal_demo_replaced_by_001_rotation", "2_random_goal_demo_replaced_by_001_rotation",
         "3_random_goal_demo_replaced_by_001_rotation"]

tag_12 = ["0_1_random_goal_demo_replaced_by_informative_001_translation", "0_2_random_goal_demo_replaced_by_informative_001_translation",
         "0_3_random_goal_demo_replaced_by_informative_001_translation"]
tag_13 = ["1_random_goal_demo_replaced_by_informative_001_rotation", "2_random_goal_demo_replaced_by_informative_001_rotation",
         "3_random_goal_demo_replaced_by_informative_001_rotation"]

tag_14 = ["0_1_random_goal_demo_with_001_translation", "0_2_random_goal_demo_with_001_translation",
         "0_3_random_goal_demo_with_001_translation"]
tag_15 = ["1_random_goal_demo_with_001_rotation", "2_random_goal_demo_with_001_rotation",
         "3_random_goal_demo_with_001_rotation"]

tag_16 = ["0_1_random_goal_demo_with_translation_regularization_2_artifical_samples",
          "0_2_random_goal_demo_with_translation_regularization_2_artifical_samples",
          "0_3_random_goal_demo_with_translation_regularization_2_artifical_samples"]
tag_17 = ["0_1_random_goal_demo_with_rotation_regularization_2_artifical_samples",
          "0_2_random_goal_demo_with_rotation_regularization_2_artifical_samples",
          "0_3_random_goal_demo_with_rotation_regularization_2_artifical_samples"]

tag_18 = ["0_1_random_goal_demo_with_01_translation", "0_2_random_goal_demo_with_01_translation",
          "0_3_random_goal_demo_with_01_translation"]

tag_19 = ["0_1_random_goal_demo_larger_workspace", "0_2_random_goal_demo_larger_workspace", "0_3_random_goal_demo_larger_workspace"]
tag_20 = ["0_1_random_goal_demo_larger_workspace_with_01_translation", "0_2_random_goal_demo_larger_workspace_with_01_translation",
          "0_3_random_goal_demo_larger_workspace_with_01_translation"]

tag_21 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air", "0_2_random_goal_demo_exclude_demo_egg_in_the_air",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air"]

tag_22 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size"]
tag_23 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size"]
tag_24 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss",
          "0_2_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss"]

tag_25 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss",
          "0_2_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss"]

tag_26 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air",
          "0_2_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air",
          "0_3_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air"]

tag_27 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization",
          "2_2_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization",
          "3_3_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization"]

tag_28 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo"]


# underarm
# tag_9 = ["1_random_goal_demo_her_type_4_segment", "2_random_goal_demo_her_type_4_segment",
#          "3_random_goal_demo_her_type_4_segment"]
# tag_10 = ["1_random_goal_demo_her_type_5_segment", "2_random_goal_demo_her_type_5_segment",
#          "3_random_goal_demo_her_type_5_segment"]
# tag_4 = ["1_random_goal_demo_with_0001_translation", "2_random_goal_demo_with_0001_translation",
#          "3_random_goal_demo_with_0001_translation"]
# tag_5 = ["1_random_goal_demo_with_0001_rotation", "2_random_goal_demo_with_0001_rotation",
#          "3_random_goal_demo_with_0001_rotation"]
# tag_25 = ["1_random_goal_demo_replaced_by_informative_001_translation", "2_random_goal_demo_replaced_by_informative_001_translation"]
# tag_12 = ["1_random_goal_demo_her_type_4_full_trajectory_pd_init_08", "2_random_goal_demo_her_type_4_full_trajectory_pd_init_08",
#          "3_random_goal_demo_her_type_4_full_trajectory_pd_init_08"]
# tag_13 = ["1_random_goal_demo_her_type_5_full_trajectory_pd_init_08", "2_random_goal_demo_her_type_5_full_trajectory_pd_init_08",
#          "3_random_goal_demo_her_type_5_full_trajectory_pd_init_08"]
# tag_17 = ["1_random_goal_demo_her_type_4_full_trajectory_start_1e6", "2_random_goal_demo_her_type_4_full_trajectory_start_1e6",
#           "3_random_goal_demo_her_type_4_full_trajectory_start_1e6"]
# tag_18 = ["1_random_goal_demo_her_type_4_full_trajectory_start_2e6", "2_random_goal_demo_her_type_4_full_trajectory_start_2e6",
#           "3_random_goal_demo_her_type_4_full_trajectory_start_2e6"]
# tag_21 = ["1_random_goal_demo_her_type_4_full_trajectory_sparse_reward", "2_random_goal_demo_her_type_4_full_trajectory_sparse_reward",
#           "3_random_goal_demo_her_type_4_full_trajectory_sparse_reward"]
# tag_22 = ["1_random_goal_demo_sparse_reward", "2_random_goal_demo_sparse_reward", "3_random_goal_demo_sparse_reward"]
# tag_27 = ["1_random_goal_demo_her_type_1_goal_from_demo", "2_random_goal_demo_her_type_1_goal_from_demo",
#           "3_random_goal_demo_her_type_1_goal_from_demo"]
# tag_28 = ["1_random_goal_demo_her_type_2_goal_from_demo", "2_random_goal_demo_her_type_2_goal_from_demo",
#           "3_random_goal_demo_her_type_2_goal_from_demo"]
# # overarm
# tag_7 = ["1_random_goal_demo_with_informative_001_translation", "2_random_goal_demo_with_informative_001_translation",
#          "3_random_goal_demo_with_informative_001_translation"]
# tag_11 = ["1_random_goal_demo_with_informative_001_rotation", "2_random_goal_demo_with_informative_001_rotation",
#          "3_random_goal_demo_with_informative_001_rotation"]
# tag_15 = ["1_random_goal_demo_replaced_by_informative_001_translation", "2_random_goal_demo_replaced_by_informative_001_translation",
#          "3_random_goal_demo_replaced_by_informative_001_translation"]
# tag_19 = ["1_random_goal_demo_replaced_by_informative_001_translation_include_original_sample",
#           "2_random_goal_demo_replaced_by_informative_001_translation_include_original_sample",
#           "3_random_goal_demo_replaced_by_informative_001_translation_include_original_sample"]
# tag_20 = ["1_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample",
#           "2_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample",
#           "3_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample"]
# tag_33 = ["1_random_goal_demo_with_translation_regularization_new", "2_random_goal_demo_with_translation_regularization_new",
#           "3_random_goal_demo_with_translation_regularization_new"]
# tag_34 = ["1_random_goal_demo_with_rotation_regularization_new", "2_random_goal_demo_with_rotation_regularization_new",
#           "3_random_goal_demo_with_rotation_regularization_new"]


def plot_all_fig(prefix=underarm_prefix, tag=tag_2):
    fig, axs = plt.subplots(2, 1)
    data_list = []
    min_len = np.inf
    for i in range(len(tag)):
        data = np.load(prefix + tag[i] + ".npy")
        data_list.append(data)
        min_len = min(min_len, data.shape[0])

        axs[0].plot(range(len(data)), data, label=tag[i])
        axs[0].set_xlabel('timesteps/5000')
        axs[0].set_ylabel('average rewards')
        axs[0].legend()

    sum = np.zeros(min_len)
    for i in range(len(tag)):
        sum = sum + data_list[i][:min_len]
    average = sum/len(tag)
    axs[1].plot(range(len(average)), average, label='Average')
    axs[1].set_xlabel('timesteps/5000')
    axs[1].set_ylabel('average rewards')
    axs[1].legend()
    plt.show()


def average_over_experiments(prefix, tag):
    data_list = []
    min_len = np.inf
    for i in range(len(tag)):
        data = np.load(prefix + tag[i] + ".npy")
        data_list.append(data)
        min_len = min(min_len, data.shape[0])
    data_np = np.zeros([len(tag), min_len])
    for i in range(len(tag)):
        data_np[i, :] = np.copy(data_list[i][:min_len])
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0)
    return mean, std


def compare(prefix, tag_list, title='', label_list=[""]):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(tag_list)):
        mean, std = average_over_experiments(prefix, tag_list[i])
        # plot variance
        axs.fill_between(range(len(mean)), mean - std/math.sqrt(len(tag_list)), mean + std/math.sqrt(len(tag_list)),
                         alpha=0.4)
        # axs.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        # plot mean
        if len(label_list) == len(tag_list):
            # specify label
            axs.plot(range(len(mean)), mean, label=label_list[i])
        else:
            axs.plot(range(len(mean)), mean, label=tag_list[i][0])
    axs.set_title(prefix+title)
    axs.set_xlabel('timesteps/5000')
    axs.set_ylabel('average rewards')
    axs.legend()
    plt.show()


def compare_policy_critic(prefix, prefix_critic, tag):
    fig, axs = plt.subplots(len(tag), 1)
    for i in range(len(tag)):
        data1 = np.load(prefix + tag[i] + ".npy")
        data2 = np.load(prefix_critic + tag[i] + ".npy")
        axs[i].plot(range(len(data1)), data1, label="policy")
        axs[i].plot(range(len(data2)), data2, label="critic")
        axs[i].set_xlabel('timesteps/5000')
        axs[i].set_ylabel('average rewards')
        axs[i].legend()
    axs[0].set_title(prefix+tag[0])
    plt.show()


compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_25)
compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_26)
compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_20)
compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_27)
# baseline
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_21, tag_22, tag_23, tag_24], title="baseline")
compare(prefix=underarm_prefix, tag_list=[tag_1, tag_21, tag_24, tag_28], title="baseline")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_22, tag_23, tag_24, tag_25], title="baseline")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_24], title="baseline")
# plot_all_fig(prefix=overarm_prefix, tag=tag_1)
# plot_all_fig(prefix=overarm_prefix, tag=tag_21)
# plot_all_fig(prefix=overarm_prefix, tag=tag_24)
compare(prefix=underarm_prefix_v1, tag_list=[tag_19, tag_26, tag_20, tag_27], title="baseline")


# # translation/rotation
compare(prefix=underarm_prefix, tag_list=[tag_1, tag_10, tag_12, tag_14, tag_18], title="translation")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_11, tag_13, tag_15], title="rotation")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_10, tag_12, tag_14, tag_18], title="translation")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_11, tag_13, tag_15], title="rotation")

compare(prefix=underarm_prefix, tag_list=[tag_1, tag_5, tag_6], title="regularization")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_5, tag_6], title="regularization")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_16, tag_17], title="regularization")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_16, tag_17], title="regularization",
#         label_list=["learning from demo", "001 translation regularization averaged by 2 samples",
#                     "001 rotation regularization averaged by 2 samples"])

# # HER
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_3, tag_4], title='HER_goal_from_demo')
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_3, tag_4, tag_9], title='HER_goal_from_demo')

# plot with specific label
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_2, tag_3], title="HER")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_2, tag_3], title="HER")

# plot_all_fig(prefix=underarm_prefix, tag=tag_5)
# plot_all_fig(prefix=underarm_prefix, tag=tag_6)
# plot_all_fig(prefix=underarm_prefix, tag=tag_27)
# plot_all_fig(prefix=underarm_prefix, tag=tag_28)
# plot_all_fig(prefix=underarm_prefix, tag=tag_4)
# plot_all_fig(prefix=underarm_prefix, tag=tag_5)
# plot_all_fig(prefix=underarm_prefix, tag=tag_6)
# plot_all_fig(prefix=underarm_prefix, tag=tag_8)
# plot_all_fig(prefix=underarm_prefix, tag=tag_9)
# plot_all_fig(prefix=underarm_prefix, tag=tag_10)
# plot_all_fig(prefix=underarm_prefix, tag=tag_12)
# plot_all_fig(prefix=underarm_prefix, tag=tag_13)
# plot_all_fig(prefix=underarm_prefix, tag=tag_15)
# plot_all_fig(prefix=underarm_prefix, tag=tag_16)


# plot_all_fig(prefix=overarm_prefix, tag=tag_5)
# plot_all_fig(prefix=overarm_prefix, tag=tag_6)
# plot_all_fig(prefix=overarm_prefix, tag=tag_3)
# plot_all_fig(prefix=overarm_prefix, tag=tag_6)
# plot_all_fig(prefix=overarm_prefix, tag=tag_8)
# plot_all_fig(prefix=overarm_prefix, tag=tag_14)
# plot_all_fig(prefix=overarm_prefix, tag=tag_15)
# plot_all_fig(prefix=overarm_prefix, tag=tag_16)
# plot_all_fig(prefix=overarm_prefix, tag=tag_7)
# plot_all_fig(prefix=overarm_prefix, tag=tag_11)

