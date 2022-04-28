import numpy as np
import torch
import random
import SCDM.TD3_plus_demos.invariance as invariance
import os
import joblib

# return the concatenated state
def env_statedict_to_state(state_dict, env_name):
	# one hand
	env_list1 = ['PenSpin-v0']
	# two hands with one object
	env_list2 = ['EggCatchOverarm-v0', 'EggCatchUnderarm-v0', 'EggHandOver-v0', 'EggCatchUnderarmHard-v0',
				 'BlockCatchOverarm-v0', 'BlockCatchUnderarm-v0', 'BlockHandOver-v0',
				 'PenCatchOverarm-v0', 'PenCatchUnderarm-v0', 'PenHandOver-v0',
				 'FetchSlideDense-v1','FetchPickAndPlaceDense-v1', 'FetchPushDense-v1',
				 'FetchSlideSparse-v1','FetchPickAndPlaceSparse-v1', 'FetchPushSparse-v1']
	# two hands with two objects
	env_list3 = ['TwoEggCatchUnderArm-v0']
	if isinstance(state_dict, dict):
		if env_name in env_list1:
			state = np.copy(state_dict["observation"])
		elif env_name in env_list2:
			state = np.concatenate((state_dict["observation"],
									state_dict["desired_goal"]))
		elif env_name in env_list3:
			state = np.concatenate((state_dict["observation"],
									state_dict["desired_goal"]['object_1'],
									state_dict["desired_goal"]['object_2']))
		else:
			raise ValueError('wrong env')
	elif isinstance(state_dict, np.ndarray):
		state = np.copy(state_dict)

	return state

# process the demonstrations
class DemoProcessor():
	def __init__(self, env, demo_tag):
		self.env = env
		self.demo_tag = demo_tag
		files = os.listdir("demonstrations/demonstrations" + "_" + env + demo_tag)
		self.files = [file for file in files if file.endswith(".pkl")]

	def process(self):
		demo_states = []
		demo_prev_actions = []
		demo_goals = []
		for file in self.files:
			traj = joblib.load("demonstrations/demonstrations" + "_" + self.env + self.demo_tag + "/" + file)
			demo_states_traj= []
			demo_prev_actions_traj = []

			for k, state in enumerate(traj["sim_states"]):
				if k == 0:
					prev_action = np.zeros(traj["actions"][0].shape)
				else:
					prev_action = traj["actions"][k - 1]

				demo_states_traj.append(state)
				demo_prev_actions_traj.append(prev_action.copy())
			demo_states.append(demo_states_traj)
			demo_prev_actions.append(demo_prev_actions_traj)
			if self.env != 'PenSpin-v0':
				demo_goals.append(traj["goal"])

		return demo_states, demo_prev_actions, demo_goals


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, env_name, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.env = env_name

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.prev_action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, prev_action, done=False):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.prev_action[self.ptr] = prev_action
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.prev_action[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device),
		)

class DemoReplayBuffer(ReplayBuffer):
	def __init__(self, state_dim, action_dim, env_name, demo_tag, env_demo, max_size=int(1e6)):
		super().__init__(state_dim, action_dim, env_name, max_size)
		self.env = env_name
		self.demo_tag = demo_tag
		files = os.listdir("demonstrations/demonstrations" + "_" + env_name + demo_tag)
		self.files = [file for file in files if file.endswith(".pkl")]

		for file in self.files:
			traj = joblib.load("demonstrations/demonstrations" + "_" + self.env + self.demo_tag + "/" + file)
			#if len(traj["actions"]) != 50:
			#	continue
			env_demo.reset()
			for k, state in enumerate(traj["sim_states"]):
				prev_obs_dict = env_demo.env._get_obs()
				prev_obs = env_statedict_to_state(prev_obs_dict, env_name=self.env)

				env_demo.env.sim.set_state(state)
				if self.env == "PenSpin-v0":
					env_demo.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
					env_demo.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
				else:
					env_demo.goal = np.copy(traj["goal"])
					env_demo.env.goal = np.copy(traj["goal"])
				env_demo.env.sim.forward()

				obs_dict = env_demo.env._get_obs()
				obs = env_statedict_to_state(obs_dict, env_name=self.env)
				if k > 0:
					action = traj["actions"][k - 1]
					reward = traj["rewards"][k - 1]
					if k == 1:
						prev_action = np.zeros(traj["actions"][0].shape)
					else:
						prev_action = traj["actions"][k - 2]
					self.add(prev_obs, action, obs, reward, prev_action)


