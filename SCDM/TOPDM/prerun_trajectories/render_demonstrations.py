import os
import gym
import time
import argparse
import joblib
import dexterous_gym

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="BlockCatchOverarm-v0")
parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
args = parser.parse_args()

files = os.listdir(args.env)
experiments = [args.env+"/"+file for file in files]

# experiments = ["../../TD3_plus_demos/demonstrations/demonstrations_EggCatchUnderarm-v0_with_translation_traj/success_2.pkl"]
# experiments = ["../experiments/PenCatchUnderarm-v0_success_0.pkl"]
# experiments = ["../prerun_trajectories/"+args.env+"/BlockCatchOverarm-v0_failure_13.pkl"]
env = gym.make(args.env)

for experiment in experiments:
    traj = joblib.load(experiment)
    env.reset()
    env.env.sim.set_state(traj["sim_states"][0])
    if args.env == "PenSpin-v0":
        import numpy as np
        env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
        env.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
    else:
        env.goal = traj["goal"]
        env.env.goal = traj["goal"]
    env.render()
    for action in traj["actions"]:
        obs, reward, _, _ = env.step(action)
        time.sleep(args.delay)
        env.render()
        print(reward)
