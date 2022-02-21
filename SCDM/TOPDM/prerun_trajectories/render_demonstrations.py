import os
import gym
import time
import argparse
import joblib
import dexterous_gym
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="EggCatchOverarm-v0")
parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
args = parser.parse_args()

files = os.listdir("../prerun_trajectories/"+args.env)
experiments = ["../prerun_trajectories/"+args.env+"/"+file for file in files]

# files = os.listdir("../../TD3_plus_demos/demonstrations/demonstrations_EggCatchUnderarm-v0")
# experiments = ["../../TD3_plus_demos/demonstrations/demonstrations_EggCatchUnderarm-v0"+"/"+file for file in files]
# experiments = ["../experiments/PenCatchUnderarm-v0_success_0.pkl"]
# experiments = ["../prerun_trajectories/"+args.env+"/EggCatchOverarm-v0_success_1.pkl"]
env = gym.make(args.env)


for experiment in experiments:
    traj = joblib.load(experiment)
    env.reset()
    env.env.sim.set_state(traj["sim_states"][0])
    env.env.sim.forward()
    if args.env == "PenSpin-v0":
        import numpy as np
        env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
        env.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
    else:
        env.goal = traj["goal"]
        env.env.goal = traj["goal"]
    env.render()

    env_list1 = ["EggCatchUnderarm-v0"]
    env_list2 = ["EggCatchOverarm-v0"]
    env_list3 = ["EggCatchUnderarmHard-v0"]
    y_axis_index = 1
    hand1_mount_ind = 46
    hand1_action_ind = 0
    hand2_mount_ind = 40
    hand2_action_ind = 20
    if args.env in env_list1:
        throwing_threshold = 0.3
        catching_threshold = 0.6
        initial_pos = np.array([0.98953, 0.36191, 0.33070])
        goal_pos = traj["goal"][0:3]
    elif args.env in env_list2:
        throwing_threshold = 0.4
        catching_threshold = 1.2
        initial_pos = np.array([1, -0.2, 0.40267])
        goal_pos = traj["goal"][0:3]
    elif args.env in env_list3:
        throwing_threshold = 0.4
        catching_threshold = 1.0
        initial_pos = np.array([0.99774, 0.06903, 0.31929])
        goal_pos = traj["goal"][0:3]
    else:
        raise NotImplementedError

    obs = None
    for action in traj["actions"]:
        freeze_throwing_hand = False
        if freeze_throwing_hand:
            if obs is not None:
                dis = np.linalg.norm(obs["achieved_goal"][0:3] - goal_pos)
            if obs is not None and np.linalg.norm(obs["achieved_goal"][0:3] - initial_pos) >= throwing_threshold:
                action[hand2_mount_ind:hand2_mount_ind + 6] = np.zeros(6)
                action[hand2_action_ind:hand2_action_ind + 20] = np.zeros(20)

        freeze_catching_hand = False
        if freeze_catching_hand:
            if obs is not None:
                dis = np.linalg.norm(obs["achieved_goal"][0:3] - goal_pos)
            if obs is not None and np.linalg.norm(obs["achieved_goal"][0:3] - goal_pos) >= catching_threshold:
                hand1_mount_bias = (np.random.rand(6)-0.5)/5
                hand1_action_bias = (np.random.rand(20) - 0.5)/5
                #action[hand1_mount_ind:hand1_mount_ind + 6] += hand1_mount_bias
                action[hand1_action_ind:hand1_action_ind + 20] += hand1_action_bias
                action = np.clip(action, -1, 1)
        obs, reward, _, _ = env.step(action)
        time.sleep(args.delay)
        env.render()
        print(reward)
