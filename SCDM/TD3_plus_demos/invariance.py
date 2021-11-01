import numpy as np
from scipy.spatial.transform import Rotation as R


class Invariance():
    def __init__(self):
        pass

    def invariant_state(self, state):
        raise NotImplementedError()

    def invariant_action(self, action):
        raise NotImplementedError()

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        inv_state = self.invariant_state(state)
        inv_action = self.invariant_action(action)
        inv_next_state = self.invariant_state(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action


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


class EggCatchOverarmInvariance(Invariance):
    def __init__(self):
        super().__init__()
        # translation
        x_max_bias = 0.1
        y_max_bias = 0.1
        z_max_bias = 0.1
        self.xyz_bias = (np.random.rand(3) - 0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])

        # rotation
        zrot_max_bias = 0.05
        self.xyz_rot_bias = (np.random.rand(3)-0.5) * np.array([0, 0, zrot_max_bias])
        self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        self.bias_r_obj = R.from_rotvec(np.array([-zrot_max_bias, 0, 0]))
        self.bias_r_target = R.from_rotvec(np.array([0, -zrot_max_bias, 0]))

        self.central_point_in_global = np.array([1, 0.675, 0.15])
        self.mount_point_1_in_global = np.array([1, 1.35, 0.15])
        self.mount_point_2_in_global = np.array([1, 0, 0.15])

        self.bias_in_state_1 = np.zeros(6)
        self.bias_in_state_2 = np.zeros(6)

    def invariant_state_translation(self, state):
        inv_state = np.copy(state)

        '''
        hand_1
        '''
        # hand_1_mount (pos)
        inv_state[:3] = state[:3] + self.xyz_bias

        '''
        hand_2
        '''
        # hand_2_mount (pos)
        inv_state[60:63] = state[60:63] + np.array([-1, -1, 1]) * self.xyz_bias

        '''
        obj
        '''
        # obj_1 (pos)
        inv_state[120:123] = state[120:123] + self.xyz_bias

        # obj_1_target (pos)
        inv_state[133:136] = state[133:136] + self.xyz_bias
        return inv_state

    def invariant_action_translation(self, action):
        inv_action = np.copy(action)
        # hand_1_mount
        inv_action[46:49] = action[46:49] + self.xyz_bias * np.array([5, 5, 12.5])
        # hand_2_mount
        inv_action[40:43] = action[40:43] + self.xyz_bias * np.array([-5, -5, 12.5])

        return inv_action

    def invariant_state_rotation(self, state):
        inv_state = np.copy(state)
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
        self.bias_in_state_1[:3] = inv_state[0:3] - self.bias_r.as_matrix() @ (state[0:3]).reshape([3, 1]).reshape(-1)
        self.bias_in_state_1[3:6] = inv_state[3:6] - self.bias_r.as_matrix() @ (state[3:6]).reshape([3, 1]).reshape(-1)
        '''
        hand_2
        '''
        # hand_2_mount (position and rotation)
        # mount point position with respect to the central point
        mount_point_in_central = np.array([-1, -1, 1]) * state[60:63] + self.mount_point_2_in_global \
                                 - self.central_point_in_global
        inv_state[60:63] = np.array([-1, -1, 1]) * (
                (self.bias_r.as_matrix() @ mount_point_in_central.reshape([3, 1])).reshape(-1) +
                self.central_point_in_global - self.mount_point_2_in_global)

        inv_state[63:66] = self.bias_r.as_matrix() @ (state[63:66] + self.xyz_rot_bias).reshape([3, 1]).reshape(-1)

        # save the bias for calculating the bias in action
        self.bias_in_state_2[:3] = inv_state[60:63] - \
                                   self.bias_r.as_matrix() @ (state[60:63]).reshape([3, 1]).reshape(-1)
        self.bias_in_state_2[3:6] = inv_state[63:66] - \
                                  self.bias_r.as_matrix() @ (state[63:66]).reshape([3, 1]).reshape(-1)
        '''
        obj
        '''
        # obj (position and quaternion)
        obj_pos_in_central = state[120:123] - self.central_point_in_global
        inv_state[120:123] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))).reshape(-1) + \
                             self.central_point_in_global

        original_r = R.from_quat(state[123:127])
        inv_state[123:127] = (self.bias_r_obj * original_r).as_quat()

        # obj_target (position and quaternion)
        obj_pos_in_central = state[133:136] - self.central_point_in_global
        inv_state[133:136] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))).reshape(-1) + \
                             self.central_point_in_global

        original_r = R.from_quat(state[136:140])
        inv_state[136:140] = (self.bias_r_target * original_r).as_quat()

        return inv_state

    def invariant_action_rotation(self, action):
        inv_action = np.copy(action)

        # hand_1_mount
        inv_action[46:49] = self.bias_r.as_matrix() @ (action[46:49]).reshape([3, 1]).reshape(-1) + \
                            self.bias_in_state_1[:3] * np.array([5, 5, 12.5])
        inv_action[49:52] = self.bias_r.as_matrix() @ (action[49:52]).reshape([3, 1]).reshape(-1) + \
                            self.bias_in_state_1[3:6] * np.array([1, 5, 5])
        # hand_2_mount
        inv_action[40:43] = (self.bias_r.as_matrix() @ (action[40:43]).reshape([3, 1]).reshape(-1)) + \
                            self.bias_in_state_2[:3] * np.array([5, 5, 12.5])
        inv_action[43:46] = self.bias_r.as_matrix() @ (action[43:46]).reshape([3, 1]).reshape(-1) + \
                            self.bias_in_state_2[3:6] * np.array([1, 5, 5])

        return inv_action

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        # inv_state = self.invariant_state_translation(state)
        # inv_action = self.invariant_action_translation(action)
        # inv_next_state = self.invariant_state_translation(next_state)
        # inv_reward = reward.copy()
        # inv_prev_action = self.invariant_action_translation(prev_action)

        inv_state = self.invariant_state_rotation(state)
        inv_action = self.invariant_action_rotation(action)
        inv_next_state = self.invariant_state_rotation(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action_rotation(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action


class EggCatchUnderarmInvariance(Invariance):
    def __init__(self):
        super().__init__()
        x_max_bias = 0.03
        y_max_bias = 0.1
        z_max_bias = 0.03
        self.hand_xyz_bias = (np.random.rand(3) - 0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])
        self.obj_xyz_bias = np.array([self.hand_xyz_bias[0], self.hand_xyz_bias[2], self.hand_xyz_bias[1]])

    def invariant_state_translation(self, state):
        inv_state = np.copy(state)
        '''
        hand_1 (target in hand1)
        '''
        # hand_1_mount (pos)
        inv_state[0:3] = state[:3] + self.hand_xyz_bias

        '''
        hand_2 (throwing from hand2)
        '''
        # hand_2_mount (pos)
        inv_state[60:63] = state[60:63] + np.array([-1, 1, -1]) * self.hand_xyz_bias

        '''
        obj_1
        '''
        # obj_1 (pos)
        inv_state[120:123] = state[120:123] + np.array([-1, -1, -1]) * self.obj_xyz_bias

        # obj_1_target (pos)
        inv_state[133:136] = state[133:136] + np.array([-1, -1, -1]) * self.obj_xyz_bias
        return inv_state

    def invariant_action_translation(self, action):
        inv_action = np.copy(action)
        # hand_1_mount
        inv_action[46:49] = action[46:49] + self.hand_xyz_bias * np.array([5, 5, 12.5])
        # hand_2_mount
        inv_action[40:43] = action[40:43] + self.hand_xyz_bias * np.array([-5, 5, -12.5])

        return inv_action

    def invariant_state_rotation(self):
        raise NotImplementedError()

    def invariant_action_rotation(self, action):
        raise NotImplementedError()

    def invariant_sample_generator(self, state, action, next_state, reward, prev_action):
        inv_state = self.invariant_state_translation(state)
        inv_action = self.invariant_action_translation(action)
        inv_next_state = self.invariant_state_translation(next_state)
        inv_reward = reward.copy()
        inv_prev_action = self.invariant_action_translation(prev_action)

        return inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action
