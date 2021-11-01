import numpy as np
import matplotlib.pyplot as plt

# average_reward_1 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_with_normalizer_beta_08.npy")
# average_reward_2 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_with_normalizer_beta_09.npy")
# # average_reward_3 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_without_normalizer.npy")
# average_reward_3 = np.load("results/TD3_TwoEggCatchUnderArm-v0_0_add_symmetry_to_replay_buffer.npy")

average_reward_1 = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_her.npy")
average_reward_2 = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_her.npy")
average_reward_3 = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_normalizer.npy")
average_reward_4 = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_normalizer.npy")

average_reward_5 = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_normalizer.npy")
average_reward_6 = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_rotation_traj.npy")
# average_reward_5 = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_normalizer.npy")
average_reward_7 = np.load("results/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_translation_traj.npy")
average_reward_8 = np.load("results/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_smaller_translation_traj.npy")
# average_reward_6 = np.load("results/TD3_EggHandOver-v0_0_EggHandOver-v0_with_normalizer.npy")
# average_reward_7 = np.load("results/TD3_BlockHandOver-v0_0_BlockHandOver-v0_with_normalizer.npy")


average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")
average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")
average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")

average_reward_pen = np.load("results/TD3_PenSpin-v0_0_PenSpin.npy")


fig, axs = plt.subplots(3, 1)
# axs[0].plot(range(len(average_reward_1)), average_reward_1, label='TwoEggCatchUnderArm_with_normalizer_beta_08')
# axs[0].plot(range(len(average_reward_2)), average_reward_2, label='TwoEggCatchUnderArm_with_normalizer_beta_09')
# axs[0].plot(range(len(average_reward_3)), average_reward_3, label='TwoEggCatchUnderArm_add_symmetry')

axs[0].plot(range(len(average_reward_1)), average_reward_1, label='EggCatchOverArm_with_her')
axs[0].plot(range(len(average_reward_2)), average_reward_2, label='EggCatchUnderArm_with_her')
axs[0].plot(range(len(average_reward_3)), average_reward_3, label='EggCatchUnderArm')
axs[0].plot(range(len(average_reward_4)), average_reward_4, label='EggCatchOverArm')

axs[0].set_xlabel('timesteps/5000')
axs[0].set_ylabel('average rewards')
axs[0].legend()

axs[1].plot(range(len(average_reward_5)), average_reward_5, label='EggCatchOverArm')
axs[1].plot(range(len(average_reward_6)), average_reward_6, label='EggCatchOverArm_with_rotation')
axs[1].plot(range(len(average_reward_7)), average_reward_7, label='EggCatchOverArm_with_translation')
axs[1].plot(range(len(average_reward_8)), average_reward_8, label='EggCatchUnderArm_with_smaller_translation')

axs[1].set_xlabel('timesteps/5000')
axs[1].set_ylabel('average rewards')
axs[1].legend()

axs[2].plot(range(len(average_reward_pen)), average_reward_pen, label='Penspin')

axs[2].set_xlabel('timesteps/5000')
axs[2].set_ylabel('average rewards')
axs[2].legend()

plt.show()