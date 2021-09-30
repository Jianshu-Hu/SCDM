import numpy as np
import matplotlib.pyplot as plt

# average_reward_1 = np.load("../TD3_plus_demos/results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj.npy")
# average_reward_2 = np.load("../TD3_plus_demos/results/TD3_TwoEggCatchUnderArm-v0_0_with_symmetry_traj.npy")
average_reward_1 = np.load("../TD3_plus_demos/results/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_one_thread.npy")
average_reward_2 = np.load("../TD3_plus_demos/results/TD3_TwoEggCatchUnderArm-v0_0_with_symmetry_traj_one_thread.npy")
average_reward_3 = np.load("../TD3_plus_demos/results/TD3_TwoEggCatchUnderArm-v0_0_add_symmetry_to_replay_buffer.npy")

fig, axs = plt.subplots(1, 1)
axs.plot(range(len(average_reward_1)), average_reward_1, label='without_sym')
axs.plot(range(len(average_reward_2)), average_reward_2, label='with_sym')
axs.plot(range(len(average_reward_3)), average_reward_3, label='add_sym_to_replay_buffer')

plt.xlabel('timesteps/5000')
plt.ylabel('average rewards')
plt.legend()
plt.show()