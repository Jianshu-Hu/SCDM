import numpy as np
import matplotlib.pyplot as plt
overarm_prefix = "results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_"
underarm_prefix = "results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_"

# basic experiment
average_reward_EggCatchOverarm = np.load(overarm_prefix+"with_normalizer.npy")
average_reward_EggCatchUnderarm = np.load(underarm_prefix+"with_normalizer.npy")
average_reward_EggHandOver = np.load("results/TD3_EggHandOver-v0_0_EggHandOver-v0_with_normalizer.npy")
average_reward_BlockCatchOverarm = np.load("results/TD3_BlockCatchOverarm-v0_0_BlockCatchOverarm-v0.npy")
average_reward_BlockCatchUnderarm = np.load("results/TD3_BlockCatchUnderarm-v0_0_BlockCatchUnderarm-v0.npy")
average_reward_BlockHandOver = np.load("results/TD3_BlockHandOver-v0_0_BlockHandOver-v0_with_normalizer.npy")
average_reward_PenCatchOverarm = np.load("results/TD3_PenCatchOverarm-v0_0_PenCatchOverarm-v0.npy")
average_reward_PenCatchUnderarm = np.load("results/TD3_PenCatchUnderarm-v0_0_PenCatchUnderarm-v0.npy")
average_reward_PenHandOver = np.load("results/TD3_PenHandOver-v0_0_PenHandOver-v0.npy")

average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")

# fig, axs = plt.subplots(1, 1)
# axs.plot(range(len(average_reward_EggCatchOverarm)), average_reward_EggCatchOverarm, label='EggCatchOverArm')
# axs.plot(range(len(average_reward_EggCatchUnderarm)), average_reward_EggCatchUnderarm, label='EggCatchUnderArm')
# axs.plot(range(len(average_reward_EggHandOver)), average_reward_EggHandOver, label='EggHandOver')
# axs.plot(range(len(average_reward_BlockCatchOverarm)), average_reward_BlockCatchOverarm, label='BlockCatchOverArm')
# axs.plot(range(len(average_reward_BlockCatchUnderarm)), average_reward_BlockCatchUnderarm, label='BlockCatchUnderArm')
# axs.plot(range(len(average_reward_BlockHandOver)), average_reward_BlockHandOver, label='BlockHandOver')
# axs.plot(range(len(average_reward_PenCatchOverarm)), average_reward_PenCatchOverarm, label='PenCatchOverArm')
# axs.plot(range(len(average_reward_PenCatchUnderarm)), average_reward_PenCatchUnderarm, label='PenCatchUnderArm')
# axs.plot(range(len(average_reward_PenHandOver)), average_reward_PenHandOver, label='PenHandOver')
#
# axs.set_xlabel('timesteps/5000')
# axs.set_ylabel('average rewards')
# axs.legend()
# plt.show()

# with different goal of the demonstration
average_reward_EggCatchOverarm_true_goal_demo = np.load(overarm_prefix+"true_goal_demo.npy")
average_reward_EggCatchOverarm_noisy_goal_demo = np.load(overarm_prefix+"noisy_goal_demo.npy")
average_reward_EggCatchOverarm_random_goal_demo = np.load(overarm_prefix+"random_goal_demo.npy")
average_reward_EggCatchUnderarm_true_goal_demo = np.load(underarm_prefix+"true_goal_demo.npy")
average_reward_EggCatchUnderarm_noisy_goal_demo = np.load(underarm_prefix+"noisy_goal_demo.npy")
average_reward_EggCatchUnderarm_random_goal_demo = np.load(underarm_prefix+"random_goal_demo.npy")

# with her
average_reward_EggCatchOverarm_random_goal_demo_her_type_1_num_1 = np.load(overarm_prefix+"random_goal_demo_her_type_1_num_1.npy")
average_reward_EggCatchOverarm_random_goal_demo_her_type_2_num_1 = np.load(overarm_prefix+"random_goal_demo_her_type_2_num_1.npy")
average_reward_EggCatchOverarm_random_goal_demo_her_type_3_num_1 = np.load(overarm_prefix+"random_goal_demo_her_type_3_num_1.npy")


average_reward_EggCatchUnderarm_random_goal_demo_her_type_1_num_1 = np.load(underarm_prefix+"random_goal_demo_her_type_1_num_1.npy")
average_reward_EggCatchUnderarm_random_goal_demo_her_type_2_num_1 = np.load(underarm_prefix+"random_goal_demo_her_type_2_num_1.npy")


# with translation
average_reward_EggCatchOverarm_small_restricted_translation_segment = \
    np.load(overarm_prefix+"with_small_restricted_translation_segment.npy")
average_reward_EggCatchOverarm_informative_translation_segment = \
    np.load(overarm_prefix+"with_informative_translation_segment.npy")


average_reward_EggCatchUnderarm_small_translation_segment = \
    np.load(underarm_prefix+"with_small_translation_segment.npy")
average_reward_EggCatchUnderarm_small_restricted_translation_segment = \
    np.load(underarm_prefix+"with_small_restricted_translation_segment.npy")
average_reward_EggCatchUnderarm_informative_translation_segment = \
    np.load(underarm_prefix+"with_informative_translation_segment.npy")
average_reward_EggCatchUnderarm_smaller_restricted_translation_segment = \
    np.load(underarm_prefix+"with_smaller_restricted_translation_segment.npy")
average_reward_EggCatchUnderarm_smaller_informative_translation_segment = \
    np.load(underarm_prefix+"with_smaller_informative_translation_segment.npy")

# with rotation
average_reward_EggCatchOverarm_restricted_rotation_segment = np.load(overarm_prefix+"with_restricted_rotation_segment.npy")
average_reward_EggCatchOverarm_informative_rotation_segment = np.load(overarm_prefix+"with_informative_rotation_segment.npy")
average_reward_EggCatchUnderarm_restricted_rotation_segment = np.load(underarm_prefix+"with_restricted_rotation_segment.npy")
average_reward_EggCatchUnderarm_informative_rotation_segment = np.load(underarm_prefix+"with_informative_rotation_segment.npy")
average_reward_EggCatchUnderarm_0003_restricted_rotation_segment = np.load(underarm_prefix+"with_0003_restricted_rotation_segment.npy")
average_reward_EggCatchUnderarm_0003_informative_rotation_segment = np.load(underarm_prefix+"with_0003_informative_rotation_segment.npy")

fig, axs = plt.subplots(6, 2)
# EggCatchoverarm
axs[0, 0].plot(range(len(average_reward_EggCatchOverarm)), average_reward_EggCatchOverarm, label='EggCatchOverArm')
axs[0, 0].set_xlabel('timesteps/5000')
axs[0, 0].set_ylabel('average rewards')
axs[0, 0].legend()

axs[1, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo_her_type_1_num_1)), average_reward_EggCatchOverarm_random_goal_demo_her_type_1_num_1, label='EggCatchOverArm_random_goal_demo_her_type_1_num_1')
axs[1, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo_her_type_2_num_1)), average_reward_EggCatchOverarm_random_goal_demo_her_type_2_num_1, label='EggCatchOverArm_random_goal_demo_her_type_2_num_1')
axs[1, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo_her_type_3_num_1)), average_reward_EggCatchOverarm_random_goal_demo_her_type_3_num_1, label='EggCatchOverArm_random_goal_demo_her_type_3_num_1')
axs[1, 0].set_xlabel('timesteps/5000')
axs[1, 0].set_ylabel('average rewards')
axs[1, 0].legend()

axs[2, 0].plot(range(len(average_reward_EggCatchOverarm_true_goal_demo)), average_reward_EggCatchOverarm_true_goal_demo, label='EggCatchOverArm_true_goal_demo')
axs[2, 0].plot(range(len(average_reward_EggCatchOverarm_noisy_goal_demo)), average_reward_EggCatchOverarm_noisy_goal_demo, label='EggCatchOverArm_noisy_goal_demo')
axs[2, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo)), average_reward_EggCatchOverarm_random_goal_demo, label='EggCatchOverArm_random_goal_demo')
axs[2, 0].set_xlabel('timesteps/5000')
axs[2, 0].set_ylabel('average rewards')
axs[2, 0].legend()

axs[3, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo)), average_reward_EggCatchOverarm_random_goal_demo, label='EggCatchOverArm_random_goal_demo')
axs[3, 0].plot(range(len(average_reward_EggCatchOverarm_small_restricted_translation_segment)), average_reward_EggCatchOverarm_small_restricted_translation_segment, label='EggCatchOverArm_with_small_restricted_translation_segment')
axs[3, 0].plot(range(len(average_reward_EggCatchOverarm_informative_translation_segment)), average_reward_EggCatchOverarm_informative_translation_segment, label='EggCatchOverArm_with_informative_translation_segment')
axs[3, 0].set_xlabel('timesteps/5000')
axs[3, 0].set_ylabel('average rewards')
axs[3, 0].legend()

axs[4, 0].plot(range(len(average_reward_EggCatchOverarm_random_goal_demo)), average_reward_EggCatchOverarm_random_goal_demo, label='EggCatchOverArm_random_goal_demo')
axs[4, 0].plot(range(len(average_reward_EggCatchOverarm_restricted_rotation_segment)), average_reward_EggCatchOverarm_restricted_rotation_segment, label='EggCatchOverArm_with_restricted_rotation_segment')
axs[4, 0].plot(range(len(average_reward_EggCatchOverarm_informative_rotation_segment)), average_reward_EggCatchOverarm_informative_rotation_segment, label='EggCatchOverArm_with_informative_rotation_segment')
axs[4, 0].set_xlabel('timesteps/5000')
axs[4, 0].set_ylabel('average rewards')
axs[4, 0].legend()

axs[0, 1].plot(range(len(average_reward_EggCatchUnderarm)), average_reward_EggCatchUnderarm, label='EggCatchUnderArm')
axs[0, 1].set_xlabel('timesteps/5000')
axs[0, 1].set_ylabel('average rewards')
axs[0, 1].legend()

axs[1, 1].plot(range(len(average_reward_EggCatchUnderarm_random_goal_demo_her_type_1_num_1)), average_reward_EggCatchUnderarm_random_goal_demo_her_type_1_num_1, label='EggCatchUnderArm_random_goal_demo_her_type_1_num_1')
axs[1, 1].plot(range(len(average_reward_EggCatchUnderarm_random_goal_demo_her_type_2_num_1)), average_reward_EggCatchUnderarm_random_goal_demo_her_type_2_num_1, label='EggCatchUnderArm_random_goal_demo_her_type_2_num_1')
axs[1, 1].set_xlabel('timesteps/5000')
axs[1, 1].set_ylabel('average rewards')
axs[1, 1].legend()

axs[2, 1].plot(range(len(average_reward_EggCatchUnderarm_true_goal_demo)), average_reward_EggCatchUnderarm_true_goal_demo, label='EggCatchUnderArm_true_goal_demo')
axs[2, 1].plot(range(len(average_reward_EggCatchUnderarm_noisy_goal_demo)), average_reward_EggCatchUnderarm_noisy_goal_demo, label='EggCatchUnderArm_noisy_goal_demo')
axs[2, 1].plot(range(len(average_reward_EggCatchUnderarm_random_goal_demo)), average_reward_EggCatchUnderarm_random_goal_demo, label='EggCatchUnderArm_random_goal_demo')
axs[2, 1].set_xlabel('timesteps/5000')
axs[2, 1].set_ylabel('average rewards')
axs[2, 1].legend()

# axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_random_goal_demo)), average_reward_EggCatchUnderarm_random_goal_demo, label='EggCatchUnderArm_random_goal_demo')
# axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_small_translation_segment)), average_reward_EggCatchUnderarm_small_translation_segment, label='EggCatchUnderArm_with_small_translation_segment')
# axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_small_restricted_translation_segment)), average_reward_EggCatchUnderarm_small_restricted_translation_segment, label='EggCatchUnderArm_with_small_restricted_translation_segment')
axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_informative_translation_segment)), average_reward_EggCatchUnderarm_informative_translation_segment, label='EggCatchUnderArm_with_informative_translation_segment')
axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_smaller_restricted_translation_segment)), average_reward_EggCatchUnderarm_smaller_restricted_translation_segment, label='EggCatchUnderArm_with_smaller_restricted_translation_segment')
axs[3, 1].plot(range(len(average_reward_EggCatchUnderarm_smaller_informative_translation_segment)), average_reward_EggCatchUnderarm_smaller_informative_translation_segment, label='EggCatchUnderArm_with_smaller_informative_translation_segment')
axs[3, 1].set_xlabel('timesteps/5000')
axs[3, 1].set_ylabel('average rewards')
axs[3, 1].legend()

axs[4, 1].plot(range(len(average_reward_EggCatchUnderarm_restricted_rotation_segment)), average_reward_EggCatchUnderarm_restricted_rotation_segment, label='EggCatchUnderArm_with_restricted_rotation_segment')
axs[4, 1].plot(range(len(average_reward_EggCatchUnderarm_informative_rotation_segment)), average_reward_EggCatchUnderarm_informative_rotation_segment, label='EggCatchUnderArm_with_informative_rotation_segment')
axs[4, 1].set_xlabel('timesteps/5000')
axs[4, 1].set_ylabel('average rewards')
axs[4, 1].legend()

axs[5, 1].plot(range(len(average_reward_EggCatchUnderarm_random_goal_demo)), average_reward_EggCatchUnderarm_random_goal_demo, label='EggCatchUnderArm_random_goal_demo')
axs[5, 1].plot(range(len(average_reward_EggCatchUnderarm_0003_restricted_rotation_segment)), average_reward_EggCatchUnderarm_0003_restricted_rotation_segment, label='EggCatchUnderArm_with_0003_restricted_rotation_segment')
axs[5, 1].plot(range(len(average_reward_EggCatchUnderarm_0003_informative_rotation_segment)), average_reward_EggCatchUnderarm_0003_informative_rotation_segment, label='EggCatchUnderArm_with_0003_informative_rotation_segment')
axs[5, 1].set_xlabel('timesteps/5000')
axs[5, 1].set_ylabel('average rewards')
axs[5, 1].legend()

plt.show()