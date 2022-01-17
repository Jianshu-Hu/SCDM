import os
import gym
import time
import dexterous_gym
import numpy as np
import joblib
from copy import deepcopy
import random
from scipy.spatial.transform import Rotation as R
from SCDM.TD3_plus_demos import TD3
from SCDM.TD3_plus_demos.utils import env_statedict_to_state
import torch


class InvariantTrajGenerator():
    def __init__(self, args):
        self.delay = args.delay
        self.env_name = args.env
        self.env = gym.make(args.env)
        self.files = os.listdir("../TOPDM/prerun_trajectories/" + args.env)
        self.traj_prefix = "../TOPDM/prerun_trajectories/" + args.env + "/"
        self.invariant_prefix = "../TOPDM/invariant_trajectories/" + args.env + "/invariant_"

        # state
        self.hand1_mount_index = 0
        self.hand2_mount_index = 0
        self.obj_index = 0
        self.tar_index = 0
        # action
        self.hand1_mount_action_index = 0
        self.hand2_mount_action_index = 0

        # translation initialization
        self.x_max_bias = 0
        self.y_max_bias = 0
        self.z_max_bias = 0
        self.global_bias = (np.random.rand(3)-0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])
        self.tf_hand1_to_global = np.zeros([3, 3])
        self.tf_hand2_to_global = np.zeros([3, 3])
        self.tf_global_to_hand1 = np.zeros([3, 3])
        self.tf_global_to_hand2 = np.zeros([3, 3])

        # rotation initialization
        self.zrot_max_bias = 0
        self.zrot_bias = (random.random() - 0.5) * self.zrot_max_bias
        self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        self.central_point_in_global = np.zeros(3)
        self.hand1_mount_point_in_global = np.zeros(3)
        self.hand2_mount_point_in_global = np.zeros(3)

        self.state_action_ratio_translation = np.array([5, 5, 12.5])
        self.action_state_ratio_translation = np.array([0.2, 0.2, 0.08])
        self.state_action_ratio_rotation = np.array([1.4, 5, 5])
        self.action_state_ratio_rotation = np.array([5/7, 0.2, 0.2])

        self.action_upper_bound = 1
        self.action_lower_bound = -1

    def hand_to_global(self, coordinate_in_hand, hand_index):
        global_coordinate = np.copy(coordinate_in_hand)
        if hand_index == 1:
            global_coordinate = (self.tf_hand1_to_global @ (coordinate_in_hand.reshape([3, 1]))).reshape(-1)
        elif hand_index == 2:
            global_coordinate = (self.tf_hand2_to_global @ (coordinate_in_hand.reshape([3, 1]))).reshape(-1)
        else:
            raise NotImplementedError
        return global_coordinate

    def global_to_hand(self, global_coordinate, hand_index):
        coordinate_in_hand = np.copy(global_coordinate)
        if hand_index == 1:
            coordinate_in_hand = (self.tf_global_to_hand1 @ (global_coordinate.reshape([3, 1]))).reshape(-1)
        elif hand_index == 2:
            coordinate_in_hand = (self.tf_global_to_hand2 @ (global_coordinate.reshape([3, 1]))).reshape(-1)
        else:
            raise NotImplementedError
        return coordinate_in_hand

    def translation_inv_state(self, state):
        inv_state = np.copy(state)
        '''
        hand_1 
        '''
        inv_state[self.hand1_mount_index:self.hand1_mount_index+3] = \
            state[self.hand1_mount_index:self.hand1_mount_index+3] + self.global_to_hand(self.global_bias, hand_index=1)
        '''
        hand_2
        '''
        # hand_2_mount (pos)
        inv_state[self.hand2_mount_index:self.hand2_mount_index+3] = \
            state[self.hand2_mount_index:self.hand2_mount_index+3] + self.global_to_hand(self.global_bias, hand_index=2)
        '''
        obj_1
        '''
        # obj_1 (pos)
        inv_state[self.obj_index:self.obj_index+3] = state[self.obj_index:self.obj_index+3] + self.global_bias

        # obj_1_target (pos)
        inv_state[self.tar_index:self.tar_index+3] = state[self.tar_index:self.tar_index+3] + self.global_bias

        return inv_state

    def translation_inv_action(self, action):
        inv_action = np.copy(action)
        # hand_1_mount
        inv_action[self.hand1_mount_action_index:self.hand1_mount_action_index+3] = \
            action[self.hand1_mount_action_index:self.hand1_mount_action_index+3] + \
            self.global_to_hand(self.global_bias, hand_index=1)*self.state_action_ratio_translation
        # hand_2_mount
        inv_action[self.hand2_mount_action_index:self.hand2_mount_action_index+3] = \
            action[self.hand2_mount_action_index:self.hand2_mount_action_index+3] + \
            self.global_to_hand(self.global_bias, hand_index=2)*self.state_action_ratio_translation
        return inv_action

    def set_translation_bias(self, actions):
        x_max_bias = self.x_max_bias
        y_max_bias = self.y_max_bias
        z_max_bias = self.z_max_bias
        self.global_bias = (np.random.rand(3) - 0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])
        for t in range(len(actions)):
            for i in range(10):
                inv_action = self.translation_inv_action(actions[t])
                # check if action is within the limit
                if np.all(inv_action - self.action_lower_bound >= 0) and np.all(inv_action - self.action_upper_bound <= 0):
                    break
                # decrease the translation bias
                x_max_bias, y_max_bias, z_max_bias = self.global_bias[0], self.global_bias[1], self.global_bias[2]
                self.global_bias = (np.random.rand(3) - 0.5) * np.array([1.8, 1.8, 1.6]) * np.array([x_max_bias, y_max_bias, z_max_bias])

    # apply translation on the trajectory
    def generate_translation_inv_traj(self):
        print('---start generating symmetric trajectories---')
        traj_list = []
        traj_invariant_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_invariant = deepcopy(traj)

            # reflect success (success is the same)
            # reflect rewards (rewards are the same)
            # reflect best rewards_it

            # apply translation on the sim_states within a small range
            # self.global_bias = (np.random.rand(3)-0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])
            # restrict translation
            self.set_translation_bias(traj["actions"])
            print("translation: ", self.global_bias)
            for t in range(len(traj["sim_states"])):
                traj_invariant["sim_states"][t].qpos[:] = self.translation_inv_state(traj["sim_states"][t].qpos[:])

                # reflect actions
                if t < len(traj["actions"]):
                    traj_invariant["actions"][t] = self.translation_inv_action(traj["actions"][t])
            # reflect goal
            traj_invariant["goal"][0:3] = traj["goal"][0:3] + self.global_bias

            if self.env_name == "EggCatchOverarm":
                # reflect achieved goal
                for t in range(len(traj["ag"])):
                    traj_invariant["ag"][t][0:3] = traj["ag"][t][0:3]+self.global_bias

            traj_list.append(traj)
            traj_invariant_list.append(traj_invariant)
            joblib.dump(traj_invariant, self.invariant_prefix + file)

        print('---stop generating symmetric trajectories---')
        return traj_list, traj_invariant_list

    def rotation_inv_state(self, state):
        inv_state = np.copy(state)
        '''
        hand_1
        '''
        # hand_1_mount (position and rotation)
        mount_point_in_hand = self.hand_to_global(state[self.hand1_mount_index:self.hand1_mount_index+3], hand_index=1)
        mount_point_in_central = mount_point_in_hand + self.hand1_mount_point_in_global - self.central_point_in_global
        trans_mount_point_in_central = (self.bias_r.as_matrix()@mount_point_in_central.reshape([3, 1])).reshape(-1)
        trans_mount_point_in_hand = trans_mount_point_in_central + self.central_point_in_global - self.hand1_mount_point_in_global
        inv_state[self.hand1_mount_index:self.hand1_mount_index+3] = self.global_to_hand(trans_mount_point_in_hand, hand_index=1)

        inv_state[self.hand1_mount_index+3:self.hand1_mount_index+6] = \
            state[self.hand1_mount_index+3: self.hand1_mount_index+6]+self.global_to_hand(self.xyz_rot_bias, hand_index=1)

        '''
        hand_2
        '''
        # hand_2_mount (position and rotation)
        mount_point_in_hand = self.hand_to_global(state[self.hand2_mount_index:self.hand2_mount_index + 3],
                                                  hand_index=2)
        mount_point_in_central = mount_point_in_hand + self.hand2_mount_point_in_global - self.central_point_in_global
        trans_mount_point_in_central = (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])).reshape(-1)
        trans_mount_point_in_hand = trans_mount_point_in_central + self.central_point_in_global - self.hand2_mount_point_in_global
        inv_state[self.hand2_mount_index:self.hand2_mount_index + 3] = self.global_to_hand(trans_mount_point_in_hand,
                                                                                           hand_index=2)

        inv_state[self.hand2_mount_index + 3:self.hand2_mount_index + 6] = \
            state[self.hand2_mount_index + 3: self.hand2_mount_index + 6] + self.global_to_hand(self.xyz_rot_bias,
                                                                                                hand_index=2)
        '''
        obj
        '''
        # obj (position and quaternion)
        obj_pos_in_central = state[self.obj_index:self.obj_index+3] - self.central_point_in_global
        inv_state[self.obj_index:self.obj_index+3] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                          .reshape(-1) + self.central_point_in_global

        original_r = R.from_quat(state[self.obj_index+3:self.obj_index+7])
        inv_state[self.obj_index+3:self.obj_index+7] = (self.bias_r * original_r).as_quat()

        # obj_target (position and quaternion)
        tar_pos_in_central = state[self.tar_index:self.tar_index+3] - self.central_point_in_global
        inv_state[self.tar_index:self.tar_index+3] = (self.bias_r.as_matrix() @ (tar_pos_in_central.reshape([3, 1]))) \
                                                          .reshape(-1) + self.central_point_in_global

        original_r = R.from_quat(state[self.tar_index+3:self.tar_index+7])
        inv_state[self.tar_index+3:self.tar_index+7] = (self.bias_r * original_r).as_quat()
        return inv_state

    def rotation_inv_action(self, action):
        inv_action = np.copy(action)

        '''
        hand_1
        '''
        # hand_1_mount (position and rotation)
        mount_point_in_hand = self.hand_to_global(
            self.action_state_ratio_translation*
            action[self.hand1_mount_action_index:self.hand1_mount_action_index + 3], hand_index=1)
        mount_point_in_central = mount_point_in_hand + self.hand1_mount_point_in_global - self.central_point_in_global
        trans_mount_point_in_central = (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])).reshape(-1)
        trans_mount_point_in_hand = trans_mount_point_in_central + self.central_point_in_global - self.hand1_mount_point_in_global
        inv_action[self.hand1_mount_action_index:self.hand1_mount_action_index + 3] = \
            self.global_to_hand(trans_mount_point_in_hand, hand_index=1)*self.state_action_ratio_translation

        inv_action[self.hand1_mount_action_index + 3:self.hand1_mount_action_index + 6] = \
            (self.action_state_ratio_rotation*action[self.hand1_mount_action_index + 3: self.hand1_mount_action_index + 6] +\
            self.global_to_hand(self.xyz_rot_bias, hand_index=1))*self.state_action_ratio_rotation
        '''
        hand_2
        '''
        # hand_2_mount (position and rotation)
        mount_point_in_hand = self.hand_to_global(
            self.action_state_ratio_translation*
            action[self.hand2_mount_action_index:self.hand2_mount_action_index + 3], hand_index=2)
        mount_point_in_central = mount_point_in_hand + self.hand2_mount_point_in_global - self.central_point_in_global
        trans_mount_point_in_central = (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])).reshape(-1)
        trans_mount_point_in_hand = trans_mount_point_in_central + self.central_point_in_global - self.hand2_mount_point_in_global
        inv_action[self.hand2_mount_action_index:self.hand2_mount_action_index + 3] = \
            self.global_to_hand(trans_mount_point_in_hand, hand_index=2)*self.state_action_ratio_translation

        inv_action[self.hand2_mount_action_index + 3:self.hand2_mount_action_index + 6] = \
            (self.action_state_ratio_rotation*
             action[self.hand2_mount_action_index + 3: self.hand2_mount_action_index + 6] +\
            self.global_to_hand(self.xyz_rot_bias, hand_index=2))*self.state_action_ratio_rotation
        return inv_action

    def set_rotation_bias(self, actions):
        zrot_max_bias = self.zrot_max_bias
        self.zrot_bias = (random.random() - 0.5) * zrot_max_bias
        self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        for t in range(len(actions)):
            for i in range(10):
                inv_action = self.rotation_inv_action(actions[t])
                # check if action is within the limit
                if np.all(inv_action - self.action_lower_bound >= 0) and np.all(inv_action - self.action_upper_bound <= 0):
                    break
                # decrease the rotation bias
                zrot_max_bias = self.zrot_bias
                self.zrot_bias = (random.random() - 0.5) * zrot_max_bias
                self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
                self.bias_r = R.from_rotvec(self.xyz_rot_bias)

    # apply rotation around the z-axis across the central point on the trajectory
    def generate_rotation_inv_traj(self):
        print('---start generating invariant trajectories---')
        traj_list = []
        traj_invariant_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_invariant = deepcopy(traj)

            # reflect success (success is the same)
            # reflect rewards (rewards are the same)
            # reflect best rewards_it

            # apply rotation on the sim_states within a small range
            # self.zrot_bias = (random.random() - 0.5) * self.zrot_max_bias
            # self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
            # self.bias_r = R.from_rotvec(self.xyz_rot_bias)
            # restrict rotation
            self.set_rotation_bias(traj["actions"])
            print("rotation: ", self.zrot_bias)
            for t in range(len(traj["sim_states"])):
                traj_invariant["sim_states"][t].qpos[:] = self.rotation_inv_state(traj["sim_states"][t].qpos[:])

            # reflect actions
                if t < len(traj["actions"]):
                    traj_invariant["actions"][t] = self.rotation_inv_action(traj["actions"][t])

            # reflect goal
            obj_pos_in_central = traj["goal"][0:3] - self.central_point_in_global
            traj_invariant["goal"][0:3] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                            .reshape(-1) + self.central_point_in_global

            original_r = R.from_quat(traj["goal"][3:7])
            traj_invariant["goal"][3:7] = (self.bias_r*original_r).as_quat()
            if self.env_name == "EggCatchOverarm":
                # reflect achieved goal
                for t in range(len(traj["ag"])):
                    obj_pos_in_central = traj["ag"][t][0:3] - self.central_point_in_global
                    traj_invariant["ag"][t][0:3] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                      .reshape(-1) + self.central_point_in_global
                    original_r = R.from_quat(traj["ag"][t][3:7])
                    traj_invariant["ag"][t][3:7] = (self.bias_r*original_r).as_quat()
            traj_list.append(traj)
            traj_invariant_list.append(traj_invariant)
            joblib.dump(traj_invariant, self.invariant_prefix +file)

        print('---stop generating symmetric trajectories---')
        return traj_list, traj_invariant_list

    def inv_traj_render(self, traj, traj_reflected, render_with_simulation=True):
        # testing reflected_traj
        if render_with_simulation:
            # this flag indicates setting the initial states with the traj[0] and executing the actions
            # render original traj
            print('------render original traj with simulation------')
            self.env.reset()
            self.env.env.sim.set_state(traj["sim_states"][0])
            self.env.goal = traj["goal"]
            self.env.env.goal = traj["goal"]
            self.env.env.sim.forward()
            self.env.render()
            for action in traj["actions"]:
                self.env.step(action)
                time.sleep(self.delay)
                self.env.render()

            #  render reflected traj
            print('------render reflected traj with simulation------')
            self.env.reset()
            self.env.env.sim.set_state(traj_reflected["sim_states"][0])
            self.env.goal = traj_reflected["goal"]
            self.env.env.goal = traj_reflected["goal"]
            self.env.env.sim.forward()
            self.env.render()
            for action in traj_reflected["actions"]:
                self.env.step(action)
                time.sleep(self.delay)
                self.env.render()
        else:
            # we can only use this to debug qpos, but it is not applicable for qvel
            print('------render original traj without simulation------')
            self.env.reset()
            self.env.goal = traj["goal"]
            self.env.env.goal = traj["goal"]
            for t in range(len(traj["sim_states"])):
                self.env.env.sim.set_state(traj["sim_states"][t])
                self.env.env.sim.forward()
                self.env.render()
                time.sleep(self.delay)

            print('------render reflected traj without simulation------')
            self.env.reset()
            self.env.goal = traj_reflected["goal"]
            self.env.env.goal = traj_reflected["goal"]
            for t in range(len(traj_reflected["sim_states"])):
                self.env.env.sim.set_state(traj_reflected["sim_states"][t])
                self.env.env.sim.forward()
                self.env.render()
                time.sleep(self.delay)

    def compare_real_state_with_artificial_state(self, traj_invariant, reset_everytime=False):
        # real states: the states generated by executing actions in the simulation
        # artificial states: the states generated by the invariant_trajectory_generator
        self.env.reset()
        self.env.env.sim.set_state(traj_invariant["sim_states"][0])
        self.env.env.sim.forward()
        timestep=0
        print('------start to check the difference------')
        for action in traj_invariant["actions"]:
            if reset_everytime:
                self.env.env.sim.set_state(traj_invariant["sim_states"][timestep])
                self.env.env.sim.forward()
            self.env.step(action)
            timestep = timestep + 1
            real_state = self.env.env.sim.data.qpos
            obs = self.env.env._get_obs()
            if reset_everytime:
                print('error every step: ', np.linalg.norm(real_state - traj_invariant["sim_states"][timestep].qpos))
            else:
                print('cumulative error: ', np.linalg.norm(real_state[0:6] - traj_invariant["sim_states"][timestep].qpos[0:6]))

    def recording(self, args, traj):
        from gym.wrappers import record_video
        env = record_video.RecordVideo(gym.make(args.env), video_folder='video/', name_prefix='invariant_traj_in_sim')
        # render
        env.reset()
        env.env.sim.set_state(traj["sim_states"][0])
        env.goal = traj["goal"]
        env.env.goal = traj["goal"]
        env.render()
        for action in traj["actions"]:
            env.step(action)
            time.sleep(self.delay)
            env.render()

    def evaluate_Q_value(self, filename1, filename2, traj=None, inv_traj=None):
        traj_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_list.append(traj)
        beta=0.7
        kwargs = {
            "env_name": self.env_name,
            "state_dim": env_statedict_to_state(self.env.env._get_obs(), self.env_name).shape[0],
            "action_dim": self.env.action_space.shape[0],
            "beta": beta,
            "max_action": 1.0,
            "file_name": ""
        }
        policy1 = TD3.TD3(**kwargs)
        policy1.load(filename1)
        policy2 = TD3.TD3(**kwargs)
        policy2.load(filename2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(1, len(traj["sim_states"])-1):
            self.env.reset()
            self.env.env.sim.set_state(traj["sim_states"][i])
            self.env.goal = traj["goal"]
            self.env.env.goal = traj["goal"]
            self.env.sim.forward()
            obs = self.env.env._get_obs()
            state = env_statedict_to_state(obs, self.env_name)
            action = traj["actions"][i]
            prev_action = traj["actions"][i-1]

            # self.env.reset()
            # self.env.env.sim.set_state(inv_traj["sim_states"][i])
            # self.env.goal = inv_traj["goal"]
            # self.env.env.goal = inv_traj["goal"]
            # obs = self.env.env._get_obs()
            # inv_state = env_statedict_to_state(obs, self.env_name)
            # inv_action = inv_traj["actions"][i]
            # inv_prev_action = inv_traj["actions"][i-1]

            # hand invariance
            env_list1 = ["EggCatchUnderarm-v0", "EggCatchUnderarm-v1"]
            env_list2 = ["EggCatchOverarm-v0"]
            if self.env_name in env_list1:
                self.y_axis_index = 1
                self.throwing_threshold = 0.3
                self.catching_threshold = 0.6
                self.initial_pos = np.array([0.98953, 0.36191, 0.33070])
            elif self.env_name in env_list2:
                self.y_axis_index = 1
                self.throwing_threshold = 0.5
                self.catching_threshold = 1.2
                self.initial_pos = np.array([1, -0.2, 0.40267])
            if np.linalg.norm(state[-20:-17] - self.initial_pos) >= self.throwing_threshold:
                inv_state = np.copy(state)
                inv_prev_action = np.copy(prev_action)
                inv_action = np.copy(action)
                inv_action[40:46] = (np.random.rand(6) - 0.5) * 2
                inv_action[20:40] = (np.random.rand(20) - 0.5) * 2

                with torch.no_grad():
                    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action = torch.FloatTensor(action.reshape(1, -1)).to(device)
                    prev_action = torch.FloatTensor(prev_action.reshape(1, -1)).to(device)
                    inv_state = torch.FloatTensor(inv_state.reshape(1, -1)).to(device)
                    inv_action = torch.FloatTensor(inv_action.reshape(1, -1)).to(device)
                    inv_prev_action = torch.FloatTensor(inv_prev_action.reshape(1, -1)).to(device)
                    current_Q1, current_Q2 = policy1.critic(state, action, prev_action)
                    inv_current_Q1, inv_current_Q2 = policy1.critic(inv_state, inv_action, inv_prev_action)
                    print("loss at time: ", i)
                    print(torch.nn.functional.mse_loss(current_Q1, inv_current_Q1).item())
                    print(torch.nn.functional.mse_loss(current_Q2, inv_current_Q2).item())
                    current_Q1, current_Q2 = policy2.critic(state, action, prev_action)
                    inv_current_Q1, inv_current_Q2 = policy2.critic(inv_state, inv_action, inv_prev_action)
                    print("loss of another model at time: ", i)
                    print(torch.nn.functional.mse_loss(current_Q1, inv_current_Q1).item())
                    print(torch.nn.functional.mse_loss(current_Q2, inv_current_Q2).item())

    def evaluate_transformed_policy(self, filename, eval_episodes=10, steps=75, render=False, delay=0.03):
        beta = 0.7
        kwargs = {
            "env_name":self.env_name,
            "state_dim": env_statedict_to_state(self.env.env._get_obs(), self.env_name).shape[0],
            "action_dim": self.env.action_space.shape[0],
            "beta": beta,
            "max_action": 1.0
        }
        policy = TD3.TD3(**kwargs)
        policy.load(filename)

        self.env.seed(100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state_dict = self.env.reset()
            state = env_statedict_to_state(state_dict, self.env_name)

            global_bias_obj = state[120:123] - np.array([1, -0.2, 0.40267])
            global_bias = state[0:3] - np.array([-1.69092e-16, 1.28969e-13, -0.0983158])
            print(global_bias_obj==global_bias)

            if render:
                self.env.render()
                time.sleep(delay)
            num_steps = 0
            inv_prev_action = np.zeros((self.env.action_space.shape[0],))
            while num_steps < steps:
                state = env_statedict_to_state(state_dict, self.env_name)
                self.global_bias = -global_bias
                inv_state = self.translation_inv_state(state)

                inv_action = policy.select_action(np.array(inv_state), inv_prev_action)
                self.global_bias = global_bias
                action = self.translation_inv_action(inv_action)

                state_dict, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                    time.sleep(delay)
                inv_prev_action = inv_action.copy()
                avg_reward += reward
                num_steps += 1

        avg_reward /= eval_episodes
        print("average reward: ", avg_reward)



