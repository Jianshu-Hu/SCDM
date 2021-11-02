import numpy as np


# divide the full state and action to state and action for one hand
class Divider():
    def __init__(self, one_hand_state_dim, one_hand_action_dim):
        self.one_hand_state_dim = one_hand_state_dim
        self.one_hand_action_dim = one_hand_action_dim

    def one_hand_state(self, full_state):
        raise NotImplementedError

    def one_hand_action(self, full_action):
        # initialize
        hand1_action = np.zeros(self.one_hand_action_dim)
        hand2_action = np.zeros(self.one_hand_action_dim)

        # hand1
        hand1_action[:20] = full_action[:20]
        hand1_action[20:26] = full_action[46:52]
        # hand2
        hand2_action[:20] = full_action[20:40]
        hand2_action[20:26] = full_action[40:46]

        return hand1_action, hand2_action

    def two_hand_action(self, hand1_action, hand2_action):
        full_action = np.zeros(self.one_hand_action_dim*2)
        full_action[0:20] = hand1_action[:20]
        full_action[20:40] = hand2_action[:20]
        full_action[40:46] = hand2_action[20:26]
        full_action[46:52] = hand1_action[20:26]

        return full_action


class OneObjDivider(Divider):
    def __init__(self, one_hand_state_dim, one_hand_action_dim):
        super().__init__(one_hand_state_dim, one_hand_action_dim)
        if self.one_hand_state_dim != 80:
            raise ValueError('check state dim')
        if self.one_hand_action_dim != 26:
            raise ValueError('check action dim')

    def one_hand_state(self, full_state):
        # initialize
        hand1_state = np.zeros(self.one_hand_state_dim)
        hand2_state = np.zeros(self.one_hand_state_dim)

        # hand1
        hand1_state[:60] = full_state[:60]
        hand1_state[60:80] = full_state[120:140]
        # hand2
        hand2_state[0:60] = full_state[60:120]
        hand2_state[60:80] = full_state[120:140]

        return hand1_state, hand2_state


# TODO:implement this
class TwoObjDivider(Divider):
    def __init__(self, one_hand_state_dim, one_hand_action_dim):
        super().__init__(one_hand_state_dim, one_hand_action_dim)
        if self.one_hand_state_dim != 100:
            raise ValueError('check state dim')
        if self.one_hand_action_dim != 26:
            raise ValueError('check action dim')

