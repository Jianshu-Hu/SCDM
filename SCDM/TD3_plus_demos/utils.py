import numpy as np
import torch
import symmetry

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.prev_action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.symmetric_definition = symmetry.Symmetry()

	def add(self, state, action, next_state, reward, prev_action):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.prev_action[self.ptr] = prev_action

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def symmetric_sample_generator(self, state, action, next_state, reward, prev_action):
		sym_state = self.symmetric_definition.symmetric_state(state)
		sym_action = self.symmetric_definition.symmetric_action(action)
		sym_next_state = self.symmetric_definition.symmetric_state(next_state)
		sym_reward = np.copy(reward)
		sym_prev_action = self.symmetric_definition.symmetric_action(prev_action)
		self.add(sym_state, sym_action, sym_next_state, sym_reward, sym_prev_action)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.prev_action[ind]).to(self.device)
		)