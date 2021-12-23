import os
import gym
import argparse
import time
import dexterous_gym
import numpy as np
import joblib
from copy import deepcopy
import random
from scipy.spatial.transform import Rotation as R
from SCDM.TD3_plus_demos import TD3
from SCDM.TD3_plus_demos.main import env_statedict_to_state
import torch


class Demo_v1_Generator():
    def __init__(self, args):
        self.delay = args.delay
        self.env_name = args.env
        self.env = gym.make(args.env+"-v1")
        self.workspace_amp = args.amp
        self.files = os.listdir("../TOPDM/prerun_trajectories/" + args.env + "-v0")
        self.traj_prefix = "../TOPDM/prerun_trajectories/" + args.env + "-v0" + "/"
        self.traj_v1_prefix = "../TOPDM/v1_demos/" + args.env + "-v1/"

    def generate_demo_v1_traj(self):
        print('---start generating trajectories for env v1---')
        traj_list = []
        traj_v1_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_v1 = deepcopy(traj)

            for t in range(len(traj["actions"])):
                # reflect actions
                traj_v1["actions"][t] = traj["actions"][t]/self.workspace_amp
            traj_list.append(traj)
            traj_v1_list.append(traj_v1)
            joblib.dump(traj_v1, self.traj_v1_prefix + file)
        return traj_list, traj_v1_list

    def traj_render(self, traj):
        print('------render original traj with simulation------')
        self.env.reset()
        self.env.env.sim.set_state(traj["sim_states"][0])
        self.env.goal = traj["goal"]
        self.env.env.goal = traj["goal"]
        self.env.render()
        for action in traj["actions"]:
            self.env.step(action)
            time.sleep(self.delay)
            self.env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchUnderarm")
    parser.add_argument('--amp', type=float, default=1.26)
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    v1trajgen = Demo_v1_Generator(args)
    traj_list, traj_v1_list = v1trajgen.generate_demo_v1_traj()
    v1trajgen.traj_render(traj_v1_list[0])
