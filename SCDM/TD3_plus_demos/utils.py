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
	def __init__(self, env, demo_tag, divide_into_N_parts):
		self.env = env
		self.demo_tag = demo_tag
		files = os.listdir("demonstrations/demonstrations" + "_" + env + demo_tag)
		self.files = [file for file in files if file.endswith(".pkl")]

		self.divide_into_N_parts = divide_into_N_parts

	def process(self):
		demo_states_throw = []
		demo_prev_actions_throw = []
		demo_states_catch = []
		demo_prev_actions_catch = []
		demo_goals = []
		for file in self.files:
			traj = joblib.load("demonstrations/demonstrations" + "_" + self.env + self.demo_tag + "/" + file)
			#if len(traj["actions"]) != 50:
			#	continue
			demo_states_traj_throw = []
			demo_prev_actions_traj_throw = []
			demo_states_traj_catch = []
			demo_prev_actions_traj_catch = []
			env_list1 = ["EggCatchUnderarm-v0", "EggCatchUnderarm-v1", "EggCatchUnderarmHard-v0",
						 "EggCatchOverarm-v0"]
			if self.env in env_list1:
				initial_position = traj["sim_states"][0].qpos[60:63]
				goal_position = traj["goal"][0:3]
				y_axis_index = np.argmax(np.abs(goal_position - initial_position))
			elif self.env == "PenSpin-v0":
				self.divide_into_N_parts = 1
			else:
				raise NotImplementedError

			for k, state in enumerate(traj["sim_states"]):
				if k == 0:
					prev_action = np.zeros(traj["actions"][0].shape)
				else:
					prev_action = traj["actions"][k - 1]

				if self.divide_into_N_parts == 1:
					demo_states_traj_throw.append(state)
					demo_prev_actions_traj_throw.append(prev_action.copy())
				elif self.divide_into_N_parts == 2:
					if abs((state.qpos[60:63] - goal_position)[y_axis_index]) <= 0.15:
						demo_states_traj_catch.append(state)
						demo_prev_actions_traj_catch.append(prev_action.copy())
					else:
						demo_states_traj_throw.append(state)
						demo_prev_actions_traj_throw.append(prev_action.copy())
				elif self.divide_into_N_parts == 3:
					if abs((state.qpos[60:63] - initial_position)[y_axis_index]) <= 0.15:
						demo_states_traj_throw.append(state)
						demo_prev_actions_traj_throw.append(prev_action.copy())
					elif abs((state.qpos[60:63] - goal_position)[y_axis_index]) <= 0.15:
						demo_states_traj_catch.append(state)
						demo_prev_actions_traj_catch.append(prev_action.copy())
				else:
					raise ValueError('Wrong number of dividing the dmeos')
			demo_states_throw.append(demo_states_traj_throw)
			demo_prev_actions_throw.append(demo_prev_actions_traj_throw)
			demo_states_catch.append(demo_states_traj_catch)
			demo_prev_actions_catch.append(demo_prev_actions_traj_catch)
			if self.env != 'PenSpin-v0':
				demo_goals.append(traj["goal"])

		return demo_states_throw, demo_prev_actions_throw, demo_states_catch, demo_prev_actions_catch, demo_goals


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

		self.invariance = np.zeros((max_size, 1)).astype(int)
		env_list1 = ["EggCatchUnderarm-v0"]
		env_list2 = ["EggCatchOverarm-v0"]
		env_list3 = ["EggCatchUnderarmHard-v0"]

		self.invariance_feasible = True
		if self.env in env_list1:
			self.y_axis_index = 1
			self.throwing_threshold = 0.3
			# self.catching_threshold = 0.6
			self.catching_threshold = 0.8
			self.initial_pos = np.array([0.98953, 0.36191, 0.33070])
		elif self.env in env_list2:
			self.y_axis_index = 1
			# self.throwing_threshold = 0.5
			self.throwing_threshold = 0.6
			self.catching_threshold = 1.2
			self.initial_pos = np.array([1, -0.2, 0.40267])
		elif self.env in env_list3:
			self.y_axis_index = 1
			self.throwing_threshold = 0.5
			self.catching_threshold = 1
			self.initial_pos = np.array([0.99774, 0.06903, 0.31929])
		else:
			self.invariance_feasible = False
			print("Invariance is not implemented for this env")

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, prev_action, done=False):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.prev_action[self.ptr] = prev_action
		self.done[self.ptr] = done

		if self.invariance_feasible:
			# throwing hand2 invariance
			if np.linalg.norm(state[-20:-17] - self.initial_pos) >= self.throwing_threshold:
				self.invariance[self.ptr] = 2
			# catching hand1 invariance
			elif np.linalg.norm(state[-20:-17] - state[-7:-4]) >= self.catching_threshold:
				self.invariance[self.ptr] = 1

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def add_from_other_replay_buffer(self, state_N, action_N, next_state_N, reward_N, prev_action_N, done_N):
		for i in range(state_N.shape[0]):
			self.add(state_N[i], action_N[i], next_state_N[i], reward_N[i], prev_action_N[i], done_N)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.prev_action[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device),
			self.invariance[ind].reshape(-1),
			ind
		)


class InvariantReplayBuffer(ReplayBuffer):
	def __init__(self, state_dim, action_dim, env_name, max_size=int(1e6)):
		super().__init__(state_dim, action_dim, env_name, max_size)
		if env_name == 'TwoEggCatchUnderArm-v0':
			self.invariance_definition = invariance.TwoEggCatchUnderArmInvariance()
		elif env_name == 'EggCatchOverarm-v0' or env_name == 'EggCatchOverarm-v1' or env_name == 'EggCatchOverarm-v2'\
				or env_name == 'EggCatchOverarm-v3':
			self.invariance_definition = invariance.CatchOverarmInvariance()
		elif env_name == 'EggCatchUnderarm-v0' or env_name == "EggCatchUnderarm-v1" or env_name == "EggCatchUnderarm-v3":
			self.invariance_definition = invariance.CatchUnderarmInvariance()
		else:
			print('Invariant replay buffer is not implemented for these envs')

	def create_invariant_trajectory(self, inv_type, use_informative, policy):
		if use_informative:
			self.state, self.action, self.next_state, self.reward, self.prev_action = self.invariance_definition. \
				informative_invariant_trajectory_generator(self.state, self.action, self.next_state, self.reward,
														   self.prev_action, inv_type, policy)
		else:
			self.state, self.action, self.next_state, self.reward, self.prev_action = self.invariance_definition.\
				invariant_trajectory_generator(self.state, self.action, self.next_state, self.reward, self.prev_action,
																					  inv_type=inv_type)

	def add_inv_sample(self, state, action, next_state, reward, prev_action, inv_type):
		inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action = \
			self.invariance_definition.invariant_sample_generator(state, action, next_state,
																  reward, prev_action, inv_type)
		self.state[self.ptr] = inv_state
		self.action[self.ptr] = inv_action
		self.next_state[self.ptr] = inv_next_state
		self.reward[self.ptr] = inv_reward
		self.prev_action[self.ptr] = inv_prev_action

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample_with_ind(self, ind):

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.prev_action[ind]).to(self.device)
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


