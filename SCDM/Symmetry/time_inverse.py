import os
import gym
import time
import argparse
import joblib
import dexterous_gym
from copy import deepcopy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="TwoEggCatchUnderArm-v0")
parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
args = parser.parse_args()

files = os.listdir("../TOPDM/prerun_trajectories/"+args.env)
experiments = ["../TOPDM/prerun_trajectories/"+args.env+"/"+file for file in files]

env = gym.make(args.env)

for experiment in experiments:
    traj = joblib.load(experiment)
    traj_reflected = deepcopy(traj)

    # invert states
    for t in range(len(traj["sim_states"])):
        traj_reflected["sim_states"][t] = traj["sim_states"][len(traj["sim_states"])-t-1]
    # invert actions
    for t in range(len(traj["actions"])):
        traj_reflected["actions"][t] = traj["actions"][len(traj["actions"])-t-1]


    # # render original traj
    env.reset()
    env.env.sim.set_state(traj["sim_states"][0])
    if env == "PenSpin-v0":
        import numpy as np
        env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
        env.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
    else:
        env.goal = traj["goal"]
        env.env.goal = traj["goal"]
    env.render()
    for action in traj["actions"]:
        env.step(action)
        time.sleep(args.delay)
        env.render()

    #  render reflected traj
    env.reset()
    env.env.sim.set_state(traj_reflected["sim_states"][0])
    if env == "PenSpin-v0":
        import numpy as np
        env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
        env.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
    else:
        env.goal = traj_reflected["goal"]
        env.env.goal = traj_reflected["goal"]
    env.render()
    for action in traj_reflected["actions"]:
        env.step(action)
        time.sleep(args.delay)
        env.render()