import joblib
import gym
import argparse
import os
import time
import dexterous_gym

delay = 0.03
env_name = "BlockCatchUnderarm-v0"
env = gym.make(env_name)
tag = "traj"


def render_demonstration():
    file_names = [env_name+'_'+tag+'_'+str(index)+'.pkl' for index in range(5)]
    trajectory_files = [
        "experiments/" + file for file in file_names
    ]
    for traj in trajectory_files:
        traj = joblib.load(traj)[0]
        env.reset()
        env.env.sim.set_state(traj["sim_states"][0].x)
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
            time.sleep(delay)
            env.render()


def from_generated_traj_to_demonstration():
    print('starting transferring generated trajectories to demonstrations')
    for index in range(10):
        trajectory_files = "experiments/"+env_name + '_' + tag + '_' + str(index) + '.pkl'
        traj = joblib.load(trajectory_files)[0]
        #traj = joblib.load(trajectory_files)
        demo = {
            "sim_states": [traj["sim_states"][0].x], "actions": [], "rewards": [], "success": [], "goal": traj["goal"],
            "best_rewards_it": []
        }
        for timestep in range(len(traj['actions'])):
            demo["sim_states"].append(traj['sim_states'][timestep+1][0].x)
            demo["actions"].append(traj['actions'][timestep])
            demo["rewards"].append(traj['rewards'][timestep])
            demo["success"].append(traj['success'][timestep])
            demo["best_rewards_it"].append(traj['best_rewards_it'][timestep])
        joblib.dump((demo), "prerun_trajectories/" + env_name + '/' + env_name + "_" + tag + "_" + str(index) + ".pkl")
    print('Done')


if __name__ == "__main__":
    from_generated_traj_to_demonstration()