import numpy as np
import matplotlib.pyplot as plt

# average_reward_1 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_with_normalizer_beta_08.npy")
# average_reward_2 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_with_normalizer_beta_09.npy")
# # average_reward_3 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_without_normalizer.npy")
# average_reward_3 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_add_symmetry_to_replay_buffer.npy")

# basic experiment
average_reward_EggCatchOverarm = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_normalizer.npy")
average_reward_EggCatchUnderarm = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_normalizer.npy")
average_reward_EggHandOver = np.load("results/TD3_EggHandOver-v0_0_EggHandOver-v0_with_normalizer.npy")
average_reward_BlockCatchOverarm = np.load("results/TD3_BlockCatchOverarm-v0_0_BlockCatchOverarm-v0.npy")
average_reward_BlockCatchUnderarm = np.load("results/TD3_BlockCatchUnderarm-v0_0_BlockCatchUnderarm-v0.npy")
average_reward_BlockHandOver = np.load("results/TD3_BlockHandOver-v0_0_BlockHandOver-v0_with_normalizer.npy")
average_reward_PenCatchOverarm = np.load("results/TD3_PenCatchOverarm-v0_0_PenCatchOverarm-v0.npy")
average_reward_PenCatchUnderarm = np.load("results/TD3_PenCatchUnderarm-v0_0_PenCatchUnderarm-v0.npy")
average_reward_PenHandOver = np.load("results/TD3_PenHandOver-v0_0_PenHandOver-v0.npy")

average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")

# with her
average_reward_EggCatchOverarm_her = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_her.npy")
average_reward_EggCatchOverarm_her_exclude_demo = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_her_exclude_demo.npy")
average_reward_EggCatchUnderarm_her = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_her.npy")
average_reward_EggCatchUnderarm_her_exclude_demo = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_her_exclude_demo.npy")

# with translation
average_reward_EggCatchOverarm_translation = \
    np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_translation_traj.npy")
average_reward_EggCatchUnderarm_translation = \
    np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_translation_traj.npy")
# with rotation
average_reward_EggCatchOverarm_rotation = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_rotation_traj.npy")



fig, axs = plt.subplots(3, 1)
# EggCatchoverarm
axs[0].plot(range(len(average_reward_EggCatchOverarm)), average_reward_EggCatchOverarm, label='EggCatchOverArm')
axs[0].plot(range(len(average_reward_EggCatchOverarm_her)), average_reward_EggCatchOverarm_her, label='EggCatchOverArm_with_her')
axs[0].plot(range(len(average_reward_EggCatchOverarm_her_exclude_demo)), average_reward_EggCatchOverarm_her_exclude_demo, label='EggCatchOverArm_with_her_exclude_demo')
axs[0].plot(range(len(average_reward_EggCatchOverarm_rotation)), average_reward_EggCatchOverarm_rotation, label='EggCatchOverArm_with_rotation')
axs[0].plot(range(len(average_reward_EggCatchOverarm_translation)), average_reward_EggCatchOverarm_translation, label='EggCatchOverArm_with_translation')

axs[0].set_xlabel('timesteps/5000')
axs[0].set_ylabel('average rewards')
axs[0].legend()

# EggCatchUnderarm
axs[1].plot(range(len(average_reward_EggCatchUnderarm)), average_reward_EggCatchUnderarm, label='EggCatchUnderArm')
axs[1].plot(range(len(average_reward_EggCatchUnderarm_her)), average_reward_EggCatchUnderarm_her, label='EggCatchUnderArm_with_her')
axs[1].plot(range(len(average_reward_EggCatchUnderarm_her_exclude_demo)), average_reward_EggCatchUnderarm_her_exclude_demo, label='EggCatchUnderArm_with_her_exclude_demo')
axs[1].plot(range(len(average_reward_EggCatchUnderarm_translation)), average_reward_EggCatchUnderarm_translation, label='EggCatchUnderArm_with_translation')

axs[1].set_xlabel('timesteps/5000')
axs[1].set_ylabel('average rewards')
axs[1].legend()

# basic
axs[2].plot(range(len(average_reward_BlockCatchOverarm)), average_reward_BlockCatchOverarm, label='BlockCatchOverArm')
axs[2].plot(range(len(average_reward_BlockCatchUnderarm)), average_reward_BlockCatchUnderarm, label='BlockCatchUnderArm')
axs[2].plot(range(len(average_reward_PenCatchOverarm)), average_reward_PenCatchOverarm, label='PenCatchOverArm')
axs[2].plot(range(len(average_reward_PenCatchUnderarm)), average_reward_PenCatchUnderarm, label='PenCatchUnderArm')
axs[2].plot(range(len(average_reward_PenHandOver)), average_reward_PenHandOver, label='PenHandOver')

axs[2].set_xlabel('timesteps/5000')
axs[2].set_ylabel('average rewards')
axs[2].legend()

plt.show()