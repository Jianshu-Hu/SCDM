import random

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class Invariance():
    def __init__(self):
        self.action_upper_bound = 1
        self.action_lower_bound = -1

        # state index
        self.hand1_mount_index = 0
        self.hand2_mount_index = 0
        self.obj_index = 0
        self.tar_index = 0
        # action index
        self.hand1_mount_action_index = 0
        self.hand2_mount_action_index = 0
        self.hand1_action_index = 0
        self.hand2_action_index = 0

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

        # informative samples
        self.n_total_segments = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def invariant_state(self, state):
        raise NotImplementedError()

    def invariant_action(self, action):
        raise NotImplementedError()

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
            (self.action_state_ratio_rotation*action[self.hand1_mount_action_index + 3: self.hand1_mount_action_index + 6]
             + self.global_to_hand(self.xyz_rot_bias, hand_index=1))*self.state_action_ratio_rotation
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
            (self.action_state_ratio_rotation*action[self.hand2_mount_action_index + 3: self.hand2_mount_action_index + 6]
             + self.global_to_hand(self.xyz_rot_bias, hand_index=2))*self.state_action_ratio_rotation
        return inv_action

    def set_translation_bias(self, actions):
        x_max_bias = self.x_max_bias
        y_max_bias = self.y_max_bias
        z_max_bias = self.z_max_bias
        self.global_bias = (np.random.rand(3) - 0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])
        for t in range(actions.shape[0]):
            for i in range(10):
                inv_action = self.translation_inv_action(actions[t, :])
                # check if action is within the limit
                if np.all(inv_action - self.action_lower_bound >= 0) and np.all(inv_action - self.action_upper_bound <= 0):
                    break
                # decrease the translation bias
                x_max_bias, y_max_bias, z_max_bias = self.global_bias[0], self.global_bias[1], self.global_bias[2]
                self.global_bias = (np.random.rand(3) - 0.5) * np.array([1.8, 1.8, 1.6]) * np.array([x_max_bias, y_max_bias, z_max_bias])

    def set_rotation_bias(self, actions):
        zrot_max_bias = self.zrot_max_bias
        self.zrot_bias = (random.random() - 0.5) * zrot_max_bias
        self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        for t in range(actions.shape[0]):
            for i in range(10):
                inv_action = self.rotation_inv_action(actions[t, :])
                # check if action is within the limit
                if np.all(inv_action - self.action_lower_bound >= 0) and np.all(inv_action - self.action_upper_bound <= 0):
                    break
                # decrease the rotation bias
                zrot_max_bias = self.zrot_bias
                self.zrot_bias = (random.random() - 0.5) * zrot_max_bias
                self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
                self.bias_r = R.from_rotvec(self.xyz_rot_bias)

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action, inv_type=""):
        if inv_type == "translation":
            # randomize the translation
            # self.global_bias = (np.random.rand(3) - 0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])

            # restrict translation
            self.set_translation_bias(action.reshape(1, -1))

            inv_state = self.translation_inv_state(state)
            inv_action = self.translation_inv_action(action)
            inv_next_state = self.translation_inv_state(next_state)
            inv_reward = reward.copy()
            inv_prev_action = self.translation_inv_action(prev_action)
        elif inv_type == "rotation":
            # randomize the rotation
            # self.zrot_bias = (random.random() - 0.5) * self.zrot_max_bias
            # self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
            # self.bias_r = R.from_rotvec(self.xyz_rot_bias)

            # restrict rotation
            self.set_rotation_bias(action.reshape(1, -1))

            inv_state = self.rotation_inv_state(state)
            inv_action = self.rotation_inv_action(action)
            inv_next_state = self.rotation_inv_state(next_state)
            inv_reward = reward.copy()
            inv_prev_action = self.rotation_inv_action(prev_action)
        else:
            inv_state = self.invariant_state(state)
            inv_action = self.invariant_action(action)
            inv_next_state = self.invariant_state(next_state)
            inv_reward = reward.copy()
            inv_prev_action = self.invariant_action(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action

    def invariant_trajectory_generator(self, state_array, action_array, next_state_array, reward_array, prev_action_array,
                                       inv_type='translation'):
        inv_state_array = np.copy(state_array)
        inv_action_array = np.copy(action_array)
        inv_next_state_array = np.copy(next_state_array)
        inv_reward_array = np.copy(reward_array)
        inv_prev_action_array = np.copy(prev_action_array)
        if inv_type == "translation":
            # randomize the translation
            # self.global_bias = (np.random.rand(3) - 0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])

            # restrict translation
            self.set_translation_bias(action_array)
        elif inv_type == "rotation":
            # randomize the rotation
            # self.zrot_bias = (random.random() - 0.5) * self.zrot_max_bias
            # self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
            # self.bias_r = R.from_rotvec(self.xyz_rot_bias)

            # restrict rotation
            self.set_rotation_bias(action_array)


        for i in range(state_array.shape[0]):
            if inv_type == 'translation':
                inv_state_array[i, :] = self.translation_inv_state(state_array[i, :])
                inv_action_array[i, :] = self.translation_inv_action(action_array[i, :])
                inv_next_state_array[i, :] = self.translation_inv_state(next_state_array[i, :])
                inv_reward_array[i, :] = reward_array[i, :].copy()
                inv_prev_action_array[i, :] = self.translation_inv_action(prev_action_array[i, :])
            elif inv_type =='rotation':
                inv_state_array[i, :] = self.rotation_inv_state(state_array[i, :])
                inv_action_array[i, :] = self.rotation_inv_action(action_array[i, :])
                inv_next_state_array[i, :] = self.rotation_inv_state(next_state_array[i, :])
                inv_reward_array[i, :] = reward_array[i, :].copy()
                inv_prev_action_array[i, :] = self.rotation_inv_action(prev_action_array[i, :])
            else:
                inv_state_array[i, :], inv_action_array[i, :], inv_next_state_array[i, :], inv_reward_array[i, :], \
                    inv_prev_action_array[i, :] = self.invariant_sample_generator(state_array[i, :], action_array[i, :],
                    next_state_array[i, :], reward_array[i, :], prev_action_array[i, :])
        return inv_state_array, inv_action_array, inv_next_state_array, inv_reward_array, inv_prev_action_array

    def informative_invariant_trajectory_generator(self, state_array, action_array, next_state_array, reward_array,
                                               prev_action_array, inv_type, policy):
        best_inv_state_array = np.copy(state_array)
        best_inv_action_array = np.copy(action_array)
        best_inv_next_state_array = np.copy(next_state_array)
        best_inv_reward_array = np.copy(reward_array)
        best_inv_prev_action_array = np.copy(prev_action_array)
        max_error = 0
        # calculate the error with original trajectory
        # for t in range(action_array.shape[0]):
        #     policy_action = (policy.select_action(state_array[t], prev_action_array[t], noise=0)). \
        #         clip(self.action_lower_bound, self.action_upper_bound)
        #     max_error += np.linalg.norm(policy_action - action_array[t])

        for i in range(self.n_total_segments):
            inv_state_array, inv_action_array, inv_next_state_array, inv_reward_array, inv_prev_action_array = \
                self.invariant_trajectory_generator(state_array, action_array, next_state_array, reward_array,
                                                prev_action_array, inv_type=inv_type)
            segment_error = 0
            for t in range(action_array.shape[0]):
                policy_action = (policy.select_action(inv_state_array[t], inv_prev_action_array[t], noise=0)).\
                    clip(self.action_lower_bound, self.action_upper_bound)
                segment_error += np.linalg.norm(policy_action-inv_action_array[t])
            if segment_error >= max_error:
                max_error = segment_error
                best_inv_state_array = np.copy(inv_state_array)
                best_inv_action_array = np.copy(inv_action_array)
                best_inv_next_state_array = np.copy(inv_next_state_array)
                best_inv_reward_array = np.copy(inv_reward_array)
                best_inv_prev_action_array = np.copy(inv_prev_action_array)
        return best_inv_state_array, best_inv_action_array, best_inv_next_state_array, best_inv_reward_array, \
               best_inv_prev_action_array

    def hand_invariance_samples_generator(self, actions, all_invariance):
        invariance = all_invariance[all_invariance!=0]
        inv_actions = np.copy(actions)

        # throwing hand2 invariance
        num_invariance_2 = np.sum(invariance==2)
        mount_action_bias_2 = (np.random.rand(num_invariance_2, 6)-0.5)*2
        hand_action_bias_2 = (np.random.rand(num_invariance_2, 20)-0.5)*2
        inv_actions[invariance==2, self.hand2_mount_action_index:self.hand2_mount_action_index + 6] += mount_action_bias_2
        inv_actions[invariance==2, self.hand2_action_index:self.hand2_action_index + 20] += hand_action_bias_2

        # catching hand1 invariance
        num_invariance_1 = np.sum(invariance == 1)
        # mount_action_bias_1 = (np.random.rand(num_invariance_1, 6)-0.5)/5
        hand_action_bias_1 = (np.random.rand(num_invariance_1, 20)-0.5)/5
        # inv_actions[invariance==1, self.hand1_mount_action_index:self.hand2_mount_action_index + 6] += mount_action_bias_1
        inv_actions[invariance==1, self.hand1_action_index:self.hand1_action_index + 20] += hand_action_bias_1

        inv_actions = np.clip(inv_actions, self.action_lower_bound, self.action_upper_bound)
        return inv_actions

    def throwing_hand_invariance_samples_generator(self, actions):
        inv_actions = np.copy(actions)

        inv_actions[:, self.hand2_mount_action_index:self.hand2_mount_action_index + 6] = np.zeros((actions.shape[0], 6))
        inv_actions[:, self.hand2_action_index:self.hand2_action_index + 20] = np.zeros((actions.shape[0], 20))

        return inv_actions

    def random_one_hand_sample_generator(self, actions):
        inv_actions = np.copy(actions)

        mount_action_bias = (np.random.rand(actions.shape[0], 6)-0.5)*2
        hand_action_bias = (np.random.rand(actions.shape[0], 20)-0.5)*2

        hand_index = int(np.random.rand(1) > 0.5)+1
        if hand_index == 1:
            inv_actions[:, self.hand1_mount_action_index:self.hand1_mount_action_index + 6] = mount_action_bias
            inv_actions[:, self.hand1_action_index:self.hand1_action_index + 20] = hand_action_bias
        elif hand_index == 2:
            inv_actions[:, self.hand2_mount_action_index:self.hand2_mount_action_index + 6] = mount_action_bias
            inv_actions[:, self.hand2_action_index:self.hand2_action_index + 20] = hand_action_bias

        return inv_actions


    def hand_fixed_samples_generator(self, actions, all_invariance):
        invariance = all_invariance[all_invariance!=0]
        inv_actions = np.copy(actions)

        # throwing hand2 invariance
        num_invariance_2 = np.sum(invariance == 2)
        mount_action_2 = np.zeros((num_invariance_2, 6))
        hand_action_2 = np.zeros((num_invariance_2, 20))
        inv_actions[invariance == 2, self.hand2_mount_action_index:self.hand2_mount_action_index + 6] = mount_action_2
        inv_actions[invariance == 2, self.hand2_action_index:self.hand2_action_index + 20] = hand_action_2

        # catching hand1 invariance
        num_invariance_1 = np.sum(invariance == 1)
        # mount_action_1 = np.zeros((num_invariance_1, 6))
        hand_action_1 = np.zeros((num_invariance_1, 20))
        # inv_actions[invariance==1, self.hand1_mount_action_index:self.hand2_mount_action_index + 6] += mount_action_1
        inv_actions[invariance == 1, self.hand1_action_index:self.hand1_action_index + 20] += hand_action_1

        inv_actions = np.zeros(actions.shape)

        return inv_actions


class CatchOverarmInvariance(Invariance):
    def __init__(self):
        super().__init__()
        # state
        self.hand1_mount_index = 0
        self.hand2_mount_index = 60
        self.obj_index = 120
        self.tar_index = 133
        # action
        self.hand1_mount_action_index = 46
        self.hand2_mount_action_index = 40
        self.hand1_action_index = 0
        self.hand2_action_index = 20

        # transformation
        self.tf_hand1_to_global = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
        self.tf_hand2_to_global = np.array([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]])

        self.tf_global_to_hand1 = np.linalg.inv(self.tf_hand1_to_global)
        self.tf_global_to_hand2 = np.linalg.inv(self.tf_hand2_to_global)

        # translation max bias
        self.x_max_bias = 0.1
        self.y_max_bias = 0.1
        self.z_max_bias = 0.1

        # rotation max bias
        self.zrot_max_bias = 0.01

        # rotation info
        self.central_point_in_global = np.array([1, 0.675, 0.15])
        self.hand1_mount_point_in_global = np.array([1, 1.35, 0.15])
        self.hand2_mount_point_in_global = np.array([1, 0, 0.15])


class CatchUnderarmInvariance(Invariance):
    def __init__(self):
        super().__init__()
        # state
        self.hand1_mount_index = 0
        self.hand2_mount_index = 60
        self.obj_index = 120
        self.tar_index = 133
        # action
        self.hand1_mount_action_index = 46
        self.hand2_mount_action_index = 40
        self.hand1_action_index = 0
        self.hand2_action_index = 20

        # transformation
        self.tf_hand1_to_global = np.array([[-1, 0, 0],
                                      [0, 0, -1],
                                      [0, -1, 0]])
        self.tf_hand2_to_global = np.array([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, -1, 0]])

        self.tf_global_to_hand1 = np.linalg.inv(self.tf_hand1_to_global)
        self.tf_global_to_hand2 = np.linalg.inv(self.tf_hand2_to_global)

        # translation max bias
        self.x_max_bias = 0.1
        self.y_max_bias = 0.1
        self.z_max_bias = 0.1

        # rotation max bias
        self.zrot_max_bias = 0.01

        # rotation info
        self.central_point_in_global = np.array([1, 0.675, 0.35])
        self.hand1_mount_point_in_global = np.array([1, 1.35, 0.35])
        self.hand2_mount_point_in_global = np.array([1, 0, 0.35])


class TwoEggCatchUnderArmInvariance(Invariance):
    # Apply symmetry
    def invariant_state(self, state):
        inv_state = np.copy(state)
        '''
        hand_1
        '''
        # hand_1_mount (pos & vel)
        inv_state[:8] = state[60:68]
        inv_state[30:38] = state[90:98]
        # hand_1_finger (joints & vel)
        inv_state[8:30] = state[68:90]
        inv_state[38:60] = state[98:120]

        '''
        hand_2
        '''
        # hand_2_mount (pos & vel)
        inv_state[60:68] = state[:8]
        inv_state[90:98] = state[30:38]
        # hand_2_finger (joints & vel)
        inv_state[68:90] = state[8:30]
        inv_state[98:120] = state[38:60]

        '''
        obj_1
        '''
        # obj_1 (pos & vel)
        inv_state[120:123] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[133:136]
        inv_state[127:130] = np.array([-1, -1, 1]) * state[140:143]
        # obj_1 (rot & ang_vel)
        inv_state[123:127] = np.array([1, -1, -1, 1]) * state[136:140]
        inv_state[130:133] = np.array([-1, -1, 1]) * state[143:146]

        '''
        obj_2
        '''
        # obj_2 (pos & vel)
        inv_state[133:136] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[120:123]
        inv_state[140:143] = np.array([-1, -1, 1]) * state[127:130]
        # obj_2 (rot & ang_vel)
        inv_state[136:140] = np.array([1, -1, -1, 1]) * state[123:127]
        inv_state[143:146] = np.array([-1, -1, 1]) * state[130:133]

        '''
        desired_obj_1
        '''
        inv_state[146:149] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[153:156]
        inv_state[149:153] = np.array([1, -1, -1, 1]) * state[156:160]

        '''
        desired_obj_2
        '''
        inv_state[153:156] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[146:149]
        inv_state[156:160] = np.array([1, -1, -1, 1]) * state[149:153]

        return inv_state

    def invariant_action(self, action):
        inv_action = np.copy(action)
        # exchange joints of fingers and wrist joints
        inv_action[:20] = action[20:40]
        inv_action[20:40] = action[:20]
        # exchange pos and rot of mount
        inv_action[46:52] = action[40:46]
        inv_action[40:46] = action[46:52]
        return inv_action
