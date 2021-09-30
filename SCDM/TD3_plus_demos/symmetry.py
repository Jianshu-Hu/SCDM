import numpy as np

class Symmetry():
    def __init__(self):
        pass

    def symmetric_state(self, state):
        sym_state = np.copy(state)
        '''
        hand_1
        '''
        # hand_1_mount (pos & vel)
        sym_state[:8] = state[60:68]
        sym_state[30:38] = state[90:98]
        # hand_1_finger (joints & vel)
        sym_state[8:30] = state[68:90]
        sym_state[38:60] = state[98:120]

        '''
        hand_2
        '''
        # hand_2_mount (pos & vel)
        sym_state[60:68] = state[:8]
        sym_state[90:98] = state[30:38]
        # hand_2_finger (joints & vel)
        sym_state[68:90] = state[8:30]
        sym_state[98:120] = state[38:60]

        '''
        obj_1
        '''
        # obj_1 (pos & vel)
        sym_state[120:123] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[133:136]
        sym_state[127:130] = np.array([-1, -1, 1]) * state[140:143]
        # obj_1 (rot & ang_vel)
        sym_state[123:127] = np.array([1, -1, -1, 1]) * state[136:140]
        sym_state[130:133] = np.array([-1, -1, 1]) * state[143:146]

        '''
        obj_2
        '''
        # obj_2 (pos & vel)
        sym_state[133:136] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[120:123]
        sym_state[140:143] = np.array([-1, -1, 1]) * state[127:130]
        # obj_2 (rot & ang_vel)
        sym_state[136:140] = np.array([1, -1, -1, 1]) * state[123:127]
        sym_state[143:146] = np.array([-1, -1, 1]) * state[130:133]

        '''
        desired_obj_1
        '''
        sym_state[146:149] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[153:156]
        sym_state[149:153] = np.array([1, -1, -1, 1]) * state[156:160]

        '''
        desired_obj_2
        '''
        sym_state[153:156] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * state[146:149]
        sym_state[156:160] = np.array([1, -1, -1, 1]) * state[149:153]

        return sym_state

    def symmetric_action(self, action):
        sym_action = np.copy(action)
        # exchange joints of fingers and wrist joints
        sym_action[:20] = action[20:40]
        sym_action[20:40] = action[:20]
        # exchange pos and rot of mount
        sym_action[46:52] = action[40:46]
        sym_action[40:46] = action[46:52]
        return sym_action
