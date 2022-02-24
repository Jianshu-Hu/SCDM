import copy
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from SCDM.TD3_plus_demos.normaliser import Normaliser
import SCDM.TD3_plus_demos.invariance as invariance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim+action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state, prev_action):
		a = F.relu(self.l1(torch.cat([state, prev_action], 1)))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + 2*action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + 2*action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action, prev_action):
		sa = torch.cat([state, prev_action, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, prev_action):
		sa = torch.cat([state, prev_action, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		env_name,
		file_name,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		beta = 0.7
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.beta = beta
		self.state_dim = state_dim

		self.normaliser = Normaliser(state_dim, default_clip_range=5.0)

		self.total_it = 0
		self.save_freq = 10000
		self.file_name_critic = file_name + "_critic_loss"
		self.file_name_actor = file_name + "_actor_loss"
		self.critic_loss_saver = []
		self.actor_loss_saver = []

		self.lamb = 1
		self.lamb_decay = 0.9999995

		self.epsilon = 1
		self.epsilon_decay = 0.9999995

		self.error_threshold = 0.1

		self.env_name = env_name
		# invariant sample generator
		if env_name == 'TwoEggCatchUnderArm-v0':
			self.invariance_definition = invariance.TwoEggCatchUnderArmInvariance()
		elif env_name == 'EggCatchOverarm-v0':
			self.invariance_definition = invariance.CatchOverarmInvariance()
		elif env_name == 'EggCatchUnderarm-v0' or env_name == 'EggCatchUnderarmHard-v0':
			self.invariance_definition = invariance.CatchUnderarmInvariance()
		else:
			print('Invariance regularization is not implemented for these envs')

	def initialize_with_demo(self, demo_replay_buffer, batch_size=256, max_iteration=50000):
		# train policy
		for iter in range(max_iteration):
			# sample from demonstrations
			demo_state, demo_action, demo_next_state, demo_reward, demo_prev_action, demo_ind = \
				demo_replay_buffer.sample(batch_size)
			demo_state = torch.FloatTensor(self.normaliser.normalize(demo_state.cpu().data.numpy())).to(device)
			demo_next_state = torch.FloatTensor(self.normaliser.normalize(demo_next_state.cpu().data.numpy())).to(
				device)
			policy_action = self.beta * self.actor(demo_state, demo_prev_action) + (
						1 - self.beta) * demo_prev_action
			# behavior cloing loss
			actor_loss = F.mse_loss(policy_action, demo_action)

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			print("iter: ", iter, " loss: ", actor_loss.cpu().data.numpy())

		# train critic
		for iter in range(int(max_iteration/5)):
			# Sample from demonstrations
			demo_state, demo_action, demo_next_state, demo_reward, demo_prev_action, demo_ind = \
				demo_replay_buffer.sample(batch_size)
			demo_state = torch.FloatTensor(self.normaliser.normalize(demo_state.cpu().data.numpy())).to(device)
			demo_next_state = torch.FloatTensor(self.normaliser.normalize(demo_next_state.cpu().data.numpy())).to(device)

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
						torch.randn_like(demo_action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

				demo_next_action = self.actor_target(demo_next_state, demo_action) + noise
				demo_next_action = self.beta * demo_next_action + (1 - self.beta) * demo_action
				demo_next_action = (
					demo_next_action
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(demo_next_state, demo_next_action, demo_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = demo_reward + self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(demo_state, demo_action, demo_prev_action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			print("iter: ", iter, " loss: ", critic_loss.cpu().data.numpy())

	def select_action(self, state, prev_action, noise=None):
		state = torch.FloatTensor(self.normaliser.normalize(state.reshape(1, -1))).to(device)
		prev_action = torch.FloatTensor(prev_action.reshape(1, -1)).to(device)
		mean_ac = self.actor(state, prev_action).cpu().data.numpy().flatten()
		if noise is not None:
			mean_ac += noise
		return self.beta*mean_ac + (1-self.beta)*prev_action.cpu().data.numpy().flatten()

	def forward_with_learned_model(self, action, next_state, reward, H, transition):
		return_H = torch.clone(reward)
		for t in range(H):
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = self.actor_target(next_state, action) + noise
			next_action = self.beta * next_action + (1 - self.beta) * action
			next_action = (
				next_action
			).clamp(-self.max_action, self.max_action)

			state = torch.clone(next_state)
			next_state = transition.forward_model(state, next_action)
			action = torch.clone(next_action)
			reward = transition.compute_reward(next_state)
			return_H += self.discount**(t+1)*reward
		return return_H




	def train(self, replay_buffer, demo_replay_buffer, invariance_replay_buffer_list, transition, batch_size=100,
			  add_invariance_regularization=False, add_hand_invariance_regularization=False, add_bc_loss=False,
			  add_artificial_transitions=False):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, prev_action, all_invariance, ind = replay_buffer.sample(batch_size)

		if add_hand_invariance_regularization:
			invariance = (all_invariance!=0)
			invariant_state = state[invariance, :]
			invariant_state = torch.FloatTensor(self.normaliser.normalize(invariant_state.cpu().data.numpy())).to(device)

			invariant_action = action[invariance, :]
			new_invariant_action = torch.FloatTensor(self.invariance_definition.hand_invariance_samples_generator
													 (invariant_action.cpu().data.numpy(), all_invariance)).to(device)

			invariant_next_state = next_state[invariance, :]
			invariant_next_state = torch.FloatTensor(self.normaliser.normalize(invariant_next_state.cpu().data.numpy())).to(device)

			# invariant_reward = reward[invariance, :]
			invariant_prev_action = prev_action[invariance, :]
		state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
		next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

		add_hand_invariance_regularization_target = False
		if add_artificial_transitions:
			if self.total_it <= 10e6:
				add_transitions_type = 'MVE'
				if add_transitions_type == 'MVE':
					forward_action = 'random'
					H = 1
				elif add_transitions_type == 'ours':
					forward_action = 'invariance'
					H = 10
					filter_with_higher_target_Q = False
					filter_with_error = False
			else:
				add_artificial_transitions = False
				add_transitions_type = None
		else:
			add_transitions_type = None

		with torch.no_grad():
			if add_transitions_type == 'MVE':
				state_H = [state]
				action_H = [action]
				next_state_H = [next_state]
				reward_H = [reward]
				prev_action_H = [prev_action]
				target_Q_H = []

				for timestep in range(H):
					if forward_action == 'policy_action':
						# Select action according to policy and add clipped noise
						noise = (
								torch.randn_like(action) * self.policy_noise
						).clamp(-self.noise_clip, self.noise_clip)

						next_action = self.actor_target(next_state_H[-1], action_H[-1]) + noise
						next_action = self.beta * next_action + (1 - self.beta) * action_H[-1]
						next_action = (
							next_action
						).clamp(-self.max_action, self.max_action)
					elif forward_action == 'random':
						# random action
						next_action = 2 * (torch.rand(action.size()) - 0.5).to(device)

					new_next_state = transition.forward_model(next_state_H[-1], next_action)
					new_reward = transition.compute_reward(new_next_state)

					state_H.append(next_state_H[-1])
					prev_action_H.append(action_H[-1])
					action_H.append(next_action)
					next_state_H.append(new_next_state)
					reward_H.append(new_reward)

				noise = (
						torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

				next_action = self.actor_target(next_state_H[-1], action_H[-1]) + noise
				next_action = self.beta * next_action + (1 - self.beta) * action_H[-1]
				next_action = (
					next_action
				).clamp(-self.max_action, self.max_action)

				Q1_H, Q2_H = self.critic_target(next_state_H[-1], next_action, action_H[-1])
				Q_H = torch.min(Q1_H, Q2_H)
				target_Q = reward_H[-1] + self.discount*Q_H
				target_Q_H.append(target_Q)
				for timestep in range(H):
					if timestep == H-1:
						# for the real transition use the 1-step return instead of using N-step return
						if forward_action == 'random':
							# Select action according to policy and add clipped noise
							noise = (
									torch.randn_like(action) * self.policy_noise
							).clamp(-self.noise_clip, self.noise_clip)

							next_action = self.actor_target(next_state, action) + noise
							next_action = self.beta * next_action + (1 - self.beta) * action
							next_action = (
								next_action
							).clamp(-self.max_action, self.max_action)
							target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)
						elif forward_action == 'policy_action':
							target_Q1, target_Q2 = self.critic_target(next_state_H[0], action_H[1], action_H[0])
						target_Q = torch.min(target_Q1, target_Q2)
						target_Q = reward_H[0]+self.discount*target_Q
					else:
						target_Q = reward_H[-timestep-2]+self.discount*target_Q_H[-1]
					target_Q_H.append(target_Q)
			else:
				# Select action according to policy and add clipped noise
				noise = (
						torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

				next_action = self.actor_target(next_state, action) + noise
				next_action = self.beta * next_action + (1 - self.beta) * action
				next_action = (
					next_action
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				if add_hand_invariance_regularization_target:
					invariant_next_action = next_action[invariance, :]
					target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)
					inv_target_Q1, inv_target_Q2 = self.critic_target(invariant_next_state,
																	  invariant_next_action, new_invariant_action)
					target_Q1[invariance, :] = (target_Q1[invariance, :] + inv_target_Q1) / 2
					target_Q2[invariance, :] = (target_Q2[invariance, :] + inv_target_Q2) / 2
				else:
					target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)

				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + self.discount * target_Q

				if add_transitions_type == 'ours':
					if forward_action == 'random':
						# random action
						new_action = 2 * (torch.rand(action.size()) - 0.5).to(device)

						# # policy action
						# noise = (
						# 		torch.randn_like(action) * self.policy_noise
						# ).clamp(-self.noise_clip, self.noise_clip)
						#
						# new_action = self.actor(state, prev_action) + noise
						# new_action = self.beta * new_action + (1 - self.beta) * prev_action
						# new_action = (
						# 	new_action
						# ).clamp(-self.max_action, self.max_action)

						# # use different action
						# if np.random.rand() <= self.epsilon:
						# 	# random action
						# 	new_action = 2*(torch.rand(action.size())-0.5)
						# else:
						# 	# policy action
						# 	noise = (
						# 			torch.randn_like(action) * self.policy_noise
						# 	).clamp(-self.noise_clip, self.noise_clip)
						#
						# 	new_action = self.actor(state, prev_action) + noise
						# 	new_action = self.beta * new_action + (1 - self.beta) * prev_action
						# 	new_action = (
						# 		new_action
						# 	).clamp(-self.max_action, self.max_action)
						# self.epsilon *= self.epsilon_decay

						new_next_state = transition.forward_model(state, new_action)
						new_reward = transition.compute_reward(new_next_state)

						# Select action according to policy and add clipped noise
						noise = (
								torch.randn_like(action) * self.policy_noise
						).clamp(-self.noise_clip, self.noise_clip)

						new_next_action = self.actor_target(new_next_state, new_action) + noise
						new_next_action = self.beta * new_next_action + (1 - self.beta) * new_action
						new_next_action = (
							new_next_action
						).clamp(-self.max_action, self.max_action)

						# Compute the target Q value
						new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action, new_action)

						new_target_Q = torch.min(new_target_Q1, new_target_Q2)
						target_Q_diff = torch.abs((new_target_Q1 - new_target_Q2) / new_target_Q)
						new_target_Q = new_reward + self.discount * new_target_Q

						# filter artificial transitions with target Q
						if filter_with_higher_target_Q:
							# if self.total_it > 3e6:
							# 	filter = torch.where(new_target_Q > target_Q, 1, 0)
							# 	new_target_Q *= filter
							# filter the transition when the target Q values from two networks have a small difference
							filter_condition = torch.logical_and((target_Q_diff < self.error_threshold),
																 (new_target_Q < target_Q))
							filter = torch.where(filter_condition, 0, 1)
							new_target_Q *= filter
						elif filter_with_error:
							imagined_next_state = transition.forward_model(state, action)
							# debug
							imagined_reward = transition.compute_reward(imagined_next_state)
							reward_error = F.mse_loss(reward, imagined_reward, reduction='none')
							# Select action according to policy and add clipped noise
							noise = (
									torch.randn_like(action) * self.policy_noise
							).clamp(-self.noise_clip, self.noise_clip)

							new_next_action = self.actor_target(imagined_next_state, action) + noise
							new_next_action = self.beta * new_next_action + (1 - self.beta) * action
							new_next_action = (
								new_next_action
							).clamp(-self.max_action, self.max_action)

							# Compute the target Q value
							new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action,
																			  action)
							new_target_Q = torch.min(new_target_Q1, new_target_Q2)
							new_target_Q = imagined_reward + self.discount * new_target_Q
							target_error = F.mse_loss(new_target_Q, target_Q, reduction='none')

							error = torch.mean(F.mse_loss(imagined_next_state, next_state, reduction='none'), dim=1, keepdim=True)
							filter = torch.where(error < 0.2, 1, 0)
							new_target_Q *= filter

					elif forward_action == 'random_one_hand':
						inv_hand_action_list\
							= self.invariance_definition.invariant_one_hand_action_generator(action.cpu().data.numpy())
						# initialize
						best_action = torch.clone(inv_hand_action_list[0])
						best_target_Q = torch.ones_like(target_Q)*torch.FloatTensor([-np.inf]).to(device)
						# best_abs_diff = torch.zeros_like(target_Q)
						# best_abs_diff = torch.ones_like(target_Q)*torch.FloatTensor([np.inf]).to(device)
						for action_ind in range(len(inv_hand_action_list)):
							new_action = inv_hand_action_list[action_ind]
							new_next_state = transition.forward_model(state, new_action)
							new_reward = transition.compute_reward(new_next_state)

							# Select action according to policy and add clipped noise
							noise = (
									torch.randn_like(action) * self.policy_noise
							).clamp(-self.noise_clip, self.noise_clip)

							new_next_action = self.actor_target(new_next_state, new_action) + noise
							new_next_action = self.beta * new_next_action + (1 - self.beta) * new_action
							new_next_action = (
								new_next_action
							).clamp(-self.max_action, self.max_action)

							# Compute the target Q value
							new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action,
																			new_action)

							new_target_Q = torch.min(new_target_Q1, new_target_Q2)
							new_target_Q = new_reward + self.discount * new_target_Q

							# abs_diff = torch.abs(new_target_Q-target_Q)

							best_action = torch.where(best_target_Q < new_target_Q, new_action, best_action)
							best_target_Q = torch.where(best_target_Q < new_target_Q, new_target_Q, best_target_Q)
							# best_abs_diff = torch.where(abs_diff < best_abs_diff, abs_diff, best_abs_diff)
					elif forward_action == 'invariance':
						invariant_action_candidate = \
							self.invariance_definition.throwing_hand_random_samples_generator(action.cpu().data.numpy())
						new_next_state = transition.forward_model(state, invariant_action_candidate)
						new_reward = transition.compute_reward(new_next_state)

						# forward two states for H steps and calculate the return
						return_H_origianl = self.forward_with_learned_model(action, next_state, reward, H, transition)
						return_H_invariant = self.forward_with_learned_model(invariant_action_candidate, new_next_state,
																			 new_reward, H, transition)

						relative_diff = torch.abs((return_H_origianl - return_H_invariant) / return_H_origianl)
						invariance = torch.where(relative_diff < 0.1, 1, 0)

						noise = (
								torch.randn_like(action) * self.policy_noise
						).clamp(-self.noise_clip, self.noise_clip)

						new_next_action = self.actor_target(new_next_state, invariant_action_candidate) + noise
						new_next_action = self.beta * new_next_action + (1 - self.beta) * invariant_action_candidate
						new_next_action = (
							new_next_action
						).clamp(-self.max_action, self.max_action)

						# Compute the target Q value
						new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action,
																		  invariant_action_candidate)
						new_target_Q = torch.min(new_target_Q1, new_target_Q2)
						new_target_Q = new_reward + self.discount * new_target_Q

						new_target_Q *= invariance
		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, prev_action)
		if add_artificial_transitions:
			if add_transitions_type == 'MVE':
				current_Q1_H = []
				current_Q2_H = []
				for timestep in range(H+1):
					current_Q1, current_Q2 = self.critic(state_H[timestep], action_H[timestep], prev_action_H[timestep])
					current_Q1_H.append(current_Q1)
					current_Q2_H.append(current_Q2)
			elif add_transitions_type == 'ours':
				if forward_action == 'random':
					new_current_Q1, new_current_Q2 = self.critic(state, new_action, prev_action)
					if filter_with_higher_target_Q:
						# if self.total_it > 3e6:
						# 	new_current_Q1 *= filter
						# 	new_current_Q2 *= filter
						new_current_Q1 *= filter
						new_current_Q2 *= filter
					elif filter_with_error:
						new_current_Q1 *= filter
						new_current_Q2 *= filter
				elif forward_action == 'random_one_hand':
					new_current_Q1, new_current_Q2 = self.critic(state, best_action, prev_action)
				elif forward_action == 'invariance':
					new_current_Q1, new_current_Q2 = self.critic(state, invariant_action_candidate, prev_action)
					new_current_Q1 *= invariance
					new_current_Q2 *= invariance


		add_hand_invariance_regularization_Q = False
		add_hand_invariance_regularization_auto = False
		if add_invariance_regularization:
			sum_artificial_Q1 = torch.zeros(current_Q1.size()).to(device)
			sum_artificial_Q2 = torch.zeros(current_Q2.size()).to(device)
			for i in range(len(invariance_replay_buffer_list)):
				# invariant samples
				inv_state, inv_action, inv_next_state, inv_reward, inv_prev_action = invariance_replay_buffer_list[i]. \
					sample_with_ind(ind)
				inv_state = torch.FloatTensor(self.normaliser.normalize(inv_state.cpu().data.numpy())).to(device)
				inv_next_state = torch.FloatTensor(self.normaliser.normalize(inv_next_state.cpu().data.numpy())).to(
					device)
				# Get Q for artificial samples
				artificial_Q1, artificial_Q2 = self.critic(inv_state, inv_action, inv_prev_action)
				sum_artificial_Q1 = sum_artificial_Q1+artificial_Q1
				sum_artificial_Q2 = sum_artificial_Q2+artificial_Q2


			# Compute critic loss
			critic_loss =  F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)\
						   + F.mse_loss(current_Q1, sum_artificial_Q1/len(invariance_replay_buffer_list)) \
						   + F.mse_loss(current_Q2, sum_artificial_Q2/len(invariance_replay_buffer_list))
		elif add_hand_invariance_regularization_Q:
			Q1 = current_Q1[invariance, :]
			Q2 = current_Q2[invariance, :]
			inv_Q1, inv_Q2 = self.critic(invariant_state, new_invariant_action, invariant_prev_action)
			regularization_loss = F.mse_loss(inv_Q1, Q1)+F.mse_loss(inv_Q2, Q2)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + regularization_loss
		elif add_hand_invariance_regularization_auto:
			with torch.no_grad():
				throwing_hand_fixed_action = torch.FloatTensor(
					self.invariance_definition.throwing_hand_invariance_samples_generator(action.cpu().data.numpy()))\
					.to(device)
				inv_Q1, inv_Q2 = self.critic(state, throwing_hand_fixed_action, prev_action)
				diff = F.mse_loss(inv_Q1, current_Q1, reduction='none') + F.mse_loss(inv_Q2, current_Q2, reduction='none')
				size = torch.exp(-diff)
			regularization_loss = (size*(F.mse_loss(inv_Q1, current_Q1, reduction='none')+F.mse_loss(inv_Q2, current_Q2, reduction='none'))).mean()
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + regularization_loss
		elif add_artificial_transitions:
			if add_transitions_type == 'MVE':
				critic_loss_list = []
				for timestep in range(H+1):
					critic_loss_list.append(F.mse_loss(current_Q1_H[timestep], target_Q_H[-timestep-1]) +\
							F.mse_loss(current_Q2_H[timestep], target_Q_H[-timestep-1]))
				critic_loss = sum(critic_loss_list)/(H+1)
			elif add_transitions_type == 'ours':
				if forward_action == 'random':
					# # Compute critic loss
					# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
					# 		self.lamb*(F.mse_loss(new_current_Q1, new_target_Q)+F.mse_loss(new_current_Q2, new_target_Q))
					# self.lamb *= self.lamb_decay

					# Compute critic loss
					critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) +\
							F.mse_loss(new_current_Q1, new_target_Q) + F.mse_loss(new_current_Q2, new_target_Q)
				elif forward_action == 'random_one_hand':
					# Compute critic loss
					critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) +\
							F.mse_loss(new_current_Q1, best_target_Q) + F.mse_loss(new_current_Q2, best_target_Q)
				elif forward_action == 'invariance':
					# Compute critic loss
					critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) +\
							F.mse_loss(new_current_Q1, new_target_Q) + F.mse_loss(new_current_Q2, new_target_Q)
		else:
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_loss_saver.append(critic_loss.item())
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		# Delayed policy updates
		add_hand_invariance_regularization_policy = False
		if self.total_it % self.policy_freq == 0:
			if add_bc_loss:
				# sample from demonstrations
				demo_state, demo_action, demo_next_state, demo_reward, demo_prev_action, _, _ = \
					demo_replay_buffer.sample(int(batch_size/10))
				demo_state = torch.FloatTensor(self.normaliser.normalize(demo_state.cpu().data.numpy())).to(device)
				# demo_next_state = torch.FloatTensor(self.normaliser.normalize(demo_next_state.cpu().data.numpy())).to(device)
				policy_action = self.beta*self.actor(demo_state, demo_prev_action) + (1-self.beta)*demo_prev_action
				Q_filter=True
				if Q_filter:
					# behavior cloing loss
					behavior_cloning_loss = torch.mean(F.mse_loss(policy_action, demo_action, reduction='none'), dim=1)
					# Q-filter
					policy_Q1, policy_Q2 = self.critic(demo_state, policy_action, demo_prev_action)
					demo_Q1, demo_Q2 = self.critic(demo_state, demo_action, demo_prev_action)
					filter = torch.where(demo_Q1+demo_Q2 > policy_Q1+policy_Q2, 1, 0)
					filtered_bc_loss = torch.mean(behavior_cloning_loss*filter)
					# Compute actor losse
					actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean() \
						+ filtered_bc_loss
				else:
					# behavior cloing loss
					behavior_cloning_loss = F.mse_loss(policy_action, demo_action)
					# Compute actor losse
					actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean() \
						+ behavior_cloning_loss
			elif add_hand_invariance_regularization_policy:
				# fix the throwing hand after throwing
				new_fixed_action = torch.FloatTensor(np.zeros(invariant_action.cpu().data.numpy().shape)).to(device)
				# new_fixed_action = torch.FloatTensor(self.invariance_definition.hand_fixed_samples_generator
				# 									 (invariant_action.cpu().data.numpy(), all_invariance)).to(device)
				hand_moving_loss = F.mse_loss(invariant_action, new_fixed_action)
				actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean() \
							 + 0.1*hand_moving_loss
			else:
				# Compute actor losse
				actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean()

			self.actor_loss_saver.append(actor_loss.item())
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		if self.total_it % self.save_freq == 0:
			np.save(f"./results/{self.file_name_critic}", self.critic_loss_saver)
			np.save(f"./results/{self.file_name_actor}", self.actor_loss_saver)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		joblib.dump(self.normaliser, filename + "_normaliser")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.normaliser = joblib.load(filename+"_normaliser")
		
