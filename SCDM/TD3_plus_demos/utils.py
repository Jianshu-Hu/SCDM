import numpy as np
import torch
import SCDM.TD3_plus_demos.invariance as invariance


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, env_name, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.prev_action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		if env_name == 'TwoEggCatchUnderArm-v0':
			self.invariance_definition = invariance.TwoEggCatchUnderArmInvariance()
		elif env_name == 'EggCatchOverarm-v0':
			self.invariance_definition = invariance.EggCatchOverarmInvariance()
		elif env_name == 'EggCatchUnderarm-v0':
			self.invariance_definition = invariance.EggCatchUnderarmInvariance()
		else:
			print('Invariance is not implemented for these envs')

	def add(self, state, action, next_state, reward, prev_action, add_invariance=False):
		self.add_transition(state, action, next_state, reward, prev_action)

		if add_invariance:
			inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action = \
				self.invariance_definition.invariant_sample_generator(state, action, next_state, reward, prev_action)
			self.add_transition(inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action)

	def add_transition(self, state, action, next_state, reward, prev_action):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.prev_action[self.ptr] = prev_action

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def add_from_hindsight_replay_buffer(self, state_N, action_N, next_state_N, reward_N, prev_action_N):
		for i in range(state_N.shape[0]):
			self.add_transition(state_N[i], action_N[i], next_state_N[i], reward_N[i], prev_action_N[i])

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.prev_action[ind]).to(self.device)
		)


class HindsightReplayBuffer(ReplayBuffer):
	def __init__(self, state_dim, action_dim, env_name, segment_length, compute_reward):
		super().__init__(state_dim, action_dim, env_name, segment_length)
		# two hands with one object
		env_list2 = ['EggCatchOverarm-v0', 'EggCatchUnderarm-v0', 'EggHandOver-v0', 'BlockHandOver-v0']
		if env_name in env_list2:
			self.goal_index = 133
		else:
			raise ValueError('Wrong Env')
		self.compute_reward = compute_reward

	def choose_new_goal(self):
		# new goal is the achieved goal at the end of this segment
		new_goal = self.state[-1, self.goal_index:self.goal_index+7]
		# update new reward
		for i in range(self.max_size):
			self.reward[i] = self.compute_reward(self.state[i, self.goal_index:self.goal_index+7], new_goal, info=None)


