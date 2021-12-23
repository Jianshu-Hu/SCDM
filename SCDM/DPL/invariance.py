import random

import numpy as np
from scipy.spatial.transform import Rotation as R


class Invariance():
    def __init__(self, hand_idx):
        self.hand_idx = hand_idx

        # translation
        self.x_max_bias = 0.4
        self.y_max_bias = 0.4
        self.z_max_bias = 0.16
        self.xyz_bias = np.zeros(3)
        self.action_upper_bound = 1
        self.action_lower_bound = -1

    def invariant_state(self, state):
        raise NotImplementedError()

    def invariant_action(self, action):
        raise NotImplementedError()

    def invariant_action_translation(self, state):
        raise NotImplementedError()

    def set_translation_bias(self, action, prev_action):
        self.x_max_bias = 0.4
        self.y_max_bias = 0.4
        self.z_max_bias = 0.16
        for i in range(10):
            self.xyz_bias = (np.random.rand(3) - 0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])
            inv_action = self.invariant_action_translation(action)
            inv_prev_action = self.invariant_action_translation(prev_action)
            # check if action is within the limit
            if np.all(inv_prev_action>=self.action_lower_bound) and np.all(inv_prev_action<=self.action_upper_bound):
                if np.all(inv_action>=self.action_lower_bound) and np.all(inv_action<=self.action_upper_bound):
                    break
            # decrease the translation bias
            self.x_max_bias, self.y_max_bias, self.z_max_bias = self.xyz_bias[0], self.xyz_bias[1], self.xyz_bias[2]
        self.xyz_bias = np.zeros(3)

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        inv_state = self.invariant_state(state)
        inv_action = self.invariant_action(action)
        inv_next_state = self.invariant_state(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action


# class TwoEggCatchUnderArmInvarianceTwoPolicy(Invariance):
#     # Apply symmetry
#     # TODO:implement symmetric samples
#     # For two policies, the situation is different.
#     # It will be better to process two replay buffer at the same time, because this processing the data in
#     # replay buffer 1 needs the data in replay buffer 2 and vice versa.
#     raise NotImplemented


class EggCatchOverarmInvarianceTwoPolicy(Invariance):
    def __init__(self, hand_idx):
        super().__init__(hand_idx)
        # translation
        self.xyz_bias = (np.random.rand(3) - 0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])

        # rotation
        self.zrot_max_bias = 0.05

        self.zrot_bias = (random.random()-0.5)*self.zrot_max_bias
        self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        self.bias_r_obj = R.from_rotvec(np.array([-self.zrot_bias, 0, 0]))
        self.bias_r_target = R.from_rotvec(np.array([0, -self.zrot_bias, 0]))

        self.central_point_in_global = np.array([1, 0.675, 0.15])
        self.mount_point_1_in_global = np.array([1, 1.35, 0.15])
        self.mount_point_2_in_global = np.array([1, 0, 0.15])

        self.bias_in_state = np.zeros(6)

    def invariant_state_translation(self, state):
        inv_state = np.copy(state)
        if self.hand_idx == 1:
            '''
            hand_1
            '''
            # hand_1_mount (pos)
            inv_state[:3] = state[:3] + self.xyz_bias
        elif self.hand_idx == 2:
            '''
            hand_2
            '''
            # hand_2_mount (pos)
            inv_state[0:3] = state[0:3] + np.array([-1, -1, 1]) * self.xyz_bias
        else:
            raise ValueError('wrong hand index')

        '''
        obj
        '''
        # obj_1 (pos)
        inv_state[60:63] = state[60:63] + self.xyz_bias

        # obj_1_target (pos)
        inv_state[73:76] = state[73:76] + self.xyz_bias
        return inv_state

    def invariant_action_translation(self, action):
        inv_action = np.copy(action)
        if self.hand_idx == 1:
            # hand_1_mount
            inv_action[20:23] = action[20:23] + self.xyz_bias * np.array([5, 5, 12.5])
        elif self.hand_idx == 2:
            # hand_2_mount
            inv_action[20:23] = action[20:23] + self.xyz_bias * np.array([-5, -5, 12.5])
        else:
            raise ValueError('wrong hand index')

        return inv_action

    def invariant_state_rotation(self, state):
        inv_state = np.copy(state)
        if self.hand_idx == 1:
            '''
            hand_1
            '''
            # hand_1_mount (position and rotation)
            # mount point position with respect to the central point
            mount_point_in_central = state[0:3] + self.mount_point_1_in_global - self.central_point_in_global
            inv_state[0:3] = (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])) \
                                        .reshape(-1) + self.central_point_in_global - self.mount_point_1_in_global

            inv_state[3:6] = self.bias_r.as_matrix() @ (state[3:6] + self.xyz_rot_bias).reshape([3, 1]).reshape(-1)

            # save the bias for calculating the bias in action
            self.bias_in_state[:3] = inv_state[0:3] - self.bias_r.as_matrix() @ (state[0:3]).reshape([3, 1]).reshape(-1)
            self.bias_in_state[3:6] = inv_state[3:6] - self.bias_r.as_matrix() @ (state[3:6]).reshape([3, 1]).reshape(-1)
        elif self.hand_idx == 2:
            '''
            hand_2
            '''
            # hand_2_mount (position and rotation)
            # mount point position with respect to the central point
            mount_point_in_central = np.array([-1, -1, 1]) * state[0:3] + self.mount_point_2_in_global \
                                     - self.central_point_in_global
            inv_state[0:3] = np.array([-1, -1, 1]) * (
                    (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])).reshape(-1) +
                    self.central_point_in_global - self.mount_point_2_in_global)

            inv_state[3:6] = self.bias_r.as_matrix() @ (state[3:6] + self.xyz_rot_bias).reshape([3, 1]).reshape(-1)

            # save the bias for calculating the bias in action
            self.bias_in_state[:3] = inv_state[0:3] - \
                                       self.bias_r.as_matrix() @ (state[0:3]).reshape([3, 1]).reshape(-1)
            self.bias_in_state[3:6] = inv_state[3:6] - \
                                      self.bias_r.as_matrix() @ (state[3:6]).reshape([3, 1]).reshape(-1)
        else:
            raise ValueError('wrong hand index')
        '''
        obj
        '''
        # obj (position and quaternion)
        obj_pos_in_central = state[60:63] - self.central_point_in_global
        inv_state[60:63] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))).reshape(-1) + \
                             self.central_point_in_global

        original_r = R.from_quat(state[63:67])
        inv_state[63:67] = (self.bias_r_obj * original_r).as_quat()

        # obj_target (position and quaternion)
        obj_pos_in_central = state[73:76] - self.central_point_in_global
        inv_state[73:76] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))).reshape(-1) + \
                             self.central_point_in_global

        original_r = R.from_quat(state[76:80])
        inv_state[76:80] = (self.bias_r_target * original_r).as_quat()

        return inv_state

    def invariant_action_rotation(self, action):
        inv_action = np.copy(action)
        if self.hand_idx == 1:
            # hand_1_mount
            inv_action[20:23] = self.bias_r.as_matrix() @ (action[20:23]).reshape([3, 1]).reshape(-1) + \
                                self.bias_in_state[:3] * np.array([5, 5, 12.5])
            inv_action[23:26] = self.bias_r.as_matrix() @ (action[23:26]).reshape([3, 1]).reshape(-1) + \
                                self.bias_in_state[3:6] * np.array([1, 5, 5])
        elif self.hand_idx == 2:
            # hand_2_mount
            inv_action[20:23] = (self.bias_r.as_matrix() @ (action[20:23]).reshape([3, 1]).reshape(-1)) + \
                                self.bias_in_state[:3] * np.array([5, 5, 12.5])
            inv_action[23:26] = self.bias_r.as_matrix() @ (action[23:26]).reshape([3, 1]).reshape(-1) + \
                                self.bias_in_state[3:6] * np.array([1, 5, 5])
        else:
            raise ValueError('wrong hand index')

        return inv_action

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        # randomize translation
        self.set_translation_bias(action, prev_action)
        inv_state = self.invariant_state_translation(state)
        inv_action = self.invariant_action_translation(action)
        inv_next_state = self.invariant_state_translation(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action_translation(prev_action)

        # randomize rotation
        # self.zrot_bias = (random.random()-0.5)*self.zrot_bias
        # self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        # self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        # self.bias_r_obj = R.from_rotvec(np.array([-self.zrot_bias, 0, 0]))
        # self.bias_r_target = R.from_rotvec(np.array([0, -self.zrot_bias, 0]))
        #
        # self.central_point_in_global = np.array([1, 0.675, 0.15])
        # self.mount_point_1_in_global = np.array([1, 1.35, 0.15])
        # self.mount_point_2_in_global = np.array([1, 0, 0.15])
        #
        # self.bias_in_state = np.zeros(6)
        # inv_state = self.invariant_state_rotation(state)
        # inv_action = self.invariant_action_rotation(action)
        # inv_next_state = self.invariant_state_rotation(next_state)
        # inv_reward = reward.copy()
        # inv_prev_action = self.invariant_action_rotation(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action


class EggCatchUnderarmInvarianceTwoPolicy(Invariance):
    def __init__(self, hand_idx):
        super().__init__(hand_idx)

        self.xyz_bias = (np.random.rand(3) - 0.5) * np.array([self.x_max_bias, self.y_max_bias, self.z_max_bias])
        self.obj_xyz_bias = np.array([self.xyz_bias[0], self.xyz_bias[2], self.xyz_bias[1]])

    def invariant_state_translation(self, state):
        inv_state = np.copy(state)
        if self.hand_idx==1:
            '''
            hand_1 (target in hand1)
            '''
            # hand_1_mount (pos)
            inv_state[0:3] = state[:3] + self.xyz_bias
        elif self.hand_idx==2:
            '''
            hand_2 (throwing from hand2)
            '''
            # hand_2_mount (pos)
            inv_state[0:3] = state[0:3] + np.array([-1, 1, -1]) * self.xyz_bias
        else:
            raise ValueError('wrong hand index')

        '''
        obj_1
        '''
        # obj_1 (pos)
        inv_state[60:63] = state[60:63] + np.array([-1, -1, -1]) * self.obj_xyz_bias

        # obj_1_target (pos)
        inv_state[73:76] = state[73:76] + np.array([-1, -1, -1]) * self.obj_xyz_bias
        return inv_state

    def invariant_action_translation(self, action):
        inv_action = np.copy(action)
        if self.hand_idx == 1:
            # hand_1_mount
            inv_action[20:23] = action[20:23] + self.hand_xyz_bias * np.array([5, 5, 12.5])
        elif self.hand_idx == 2:
            # hand_2_mount
            inv_action[20:23] = action[20:23] + self.hand_xyz_bias * np.array([-5, 5, -12.5])
        else:
            raise ValueError('wrong hand index')

        return inv_action

    def invariant_state_rotation(self):
        raise NotImplementedError()

    def invariant_action_rotation(self, action):
        raise NotImplementedError()

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        # randomize translation
        self.set_translation_bias(action, prev_action)
        self.obj_xyz_bias = np.array([self.xyz_bias[0], self.xyz_bias[2], self.xyz_bias[1]])
        inv_state = self.invariant_state_translation(state)
        inv_action = self.invariant_action_translation(action)
        inv_next_state = self.invariant_state_translation(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action_translation(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action
