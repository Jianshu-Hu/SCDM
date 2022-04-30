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
		beta = 0.7,
		add_artificial_transitions_type='None'
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		if add_artificial_transitions_type == 'ours':
			# initialize with a high Q value to encourage exploration
			if env_name == 'PenSpin-v0':
				nn.init.constant_(self.critic.l3.bias.data, 50)
				nn.init.constant_(self.critic.l6.bias.data, 50)
			elif env_name == 'EggCatchOverarm-v0':
				nn.init.constant_(self.critic.l3.bias.data, 10)
				nn.init.constant_(self.critic.l6.bias.data, 10)
			elif env_name == 'EggCatchUnderarm-v0':
				nn.init.constant_(self.critic.l3.bias.data, 10)
				nn.init.constant_(self.critic.l6.bias.data, 10)
			elif env_name == 'Walker2d-v3':
				nn.init.constant_(self.critic.l3.bias.data, 50)
				nn.init.constant_(self.critic.l6.bias.data, 50)
			elif env_name == 'HalfCheetah-v3':
				nn.init.constant_(self.critic.l3.bias.data, 50)
				nn.init.constant_(self.critic.l6.bias.data, 50)
			elif env_name == 'Swimmer-v3':
				nn.init.constant_(self.critic.l3.bias.data, 20)
				nn.init.constant_(self.critic.l6.bias.data, 20)
			elif env_name == 'Hopper-v3':
				nn.init.constant_(self.critic.l3.bias.data, 50)
				nn.init.constant_(self.critic.l6.bias.data, 50)
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

		self.save_info_for_debugging = False
		self.file_name_critic = file_name + "_critic_loss"
		self.file_name_actor = file_name + "_actor_loss"
		self.critic_loss_saver = []
		self.actor_loss_saver = []

		self.file_name_variance = file_name + "_variance"
		self.variance_saver = []

		self.epsilon = 1
		self.epsilon_decay = 0.9999996

		self.noise_variance = 5
		self.noise_variance_decay = 0.9999995

		self.env_name = env_name

	def select_action(self, state, prev_action, noise=None):
		state = torch.FloatTensor(self.normaliser.normalize(state.reshape(1, -1))).to(device)
		prev_action = torch.FloatTensor(prev_action.reshape(1, -1)).to(device)
		mean_ac = self.actor(state, prev_action).cpu().data.numpy().flatten()
		if noise is not None:
			mean_ac += noise
		return self.beta*mean_ac + (1-self.beta)*prev_action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, demo_replay_buffer, transition, batch_size=100, add_bc_loss=False,
			  add_artificial_transitions_type=None, prediction_horizon=1):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, prev_action, done, = replay_buffer.sample(batch_size)

		state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
		next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

		# # debug reward
		# if self.total_it % 1000 == 0:
		# 	error = F.mse_loss(reward, transition.compute_reward(state, next_state, action))
		# 	print("reward_error: ", error)
		# # debug done
		# if self.total_it % 1000 == 0:
		# 	error = F.mse_loss(done, transition.not_healthy(next_state))
		# 	print("done_error: ", error)

		if add_artificial_transitions_type == "None":
			add_artificial_transitions = False
		else:
			add_artificial_transitions = True
			if add_artificial_transitions_type == 'MVE':
				H = prediction_horizon
			elif add_artificial_transitions_type == 'ours':
				# gaussian, uniform, fixed
				noise_type = 'gaussian'
				initial_bound = self.max_action
				max_error_so_far = torch.zeros(1)


		with torch.no_grad():
			if add_artificial_transitions_type == 'MVE':
				state_H = [state]
				action_H = [action]
				next_state_H = [next_state]
				reward_H = [reward]
				prev_action_H = [prev_action]
				done_H = [done]
				target_Q_H = []

				for timestep in range(H):
					# Select action according to policy
					next_action = self.beta * self.actor(next_state_H[-1], action_H[-1]) + \
								(1 - self.beta) * action_H[-1]
					next_action = (
						next_action
					).clamp(-self.max_action, self.max_action)

					new_next_state = transition.forward_model(next_state_H[-1], next_action)
					new_reward = transition.compute_reward(next_state_H[-1], new_next_state, next_action)
					new_done = transition.not_healthy(new_next_state)

					state_H.append(next_state_H[-1])
					prev_action_H.append(action_H[-1])
					action_H.append(next_action)
					next_state_H.append(new_next_state)
					reward_H.append(new_reward)
					done_H.append(new_done)

				# calculate the target Q at t=H
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
				target_Q = reward_H[-1] + self.discount*(1-done_H[-1])*Q_H
				target_Q_H.append(target_Q)

				# calculting the target Q backward
				for timestep in range(H):
					# if timestep == H-1:
					# 	# for the real transition use the 1-step return instead of using N-step return
					# 	target_Q1, target_Q2 = self.critic_target(next_state_H[0], action_H[1], action_H[0])
					# 	target_Q = torch.min(target_Q1, target_Q2)
					# 	target_Q = reward_H[0]+self.discount*(1-done)*target_Q
					# else:
					# 	target_Q = reward_H[-timestep-2]+self.discount*(1-done_H[-timestep-2)target_Q_H[-1]
					target_Q = reward_H[-timestep - 2] + self.discount * (1-done_H[-timestep-2]) * target_Q_H[-1]
					target_Q_H.append(target_Q)
				# remove the segment from t+1 to H when the episode ends at t before H
				filter_last = torch.ones_like(done)
				filter_last_H = [torch.ones_like(done)]
				for timestep in range(H):
					filter_last *= (1-done_H[timestep])
					target_Q_H[-timestep-2] *= filter_last
					filter_last_H.append(filter_last)

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

				target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)

				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + self.discount * (1-done) * target_Q

				if add_artificial_transitions_type == 'ours':
					# noisy policy action
					if noise_type == 'gaussian':
						# decaying clip
						noise = (
								torch.randn_like(action)
						).clamp(-self.epsilon*initial_bound, self.epsilon*initial_bound)
					elif noise_type == 'uniform':
						noise = (2*torch.rand_like(action)-torch.ones_like(action))*(self.max_action*self.epsilon)
					elif noise_type == 'fixed':
						noise = (
							torch.randn_like(action)*self.policy_noise
						).clamp(-self.noise_clip,self.noise_clip)
					self.epsilon *= self.epsilon_decay

					new_action = self.actor(state, prev_action) + noise
					new_action = self.beta * new_action + (1 - self.beta) * prev_action
					new_action = new_action.clamp(-self.max_action, self.max_action)

					# forward with the action using the dynamic model
					new_next_state = transition.forward_model(state, new_action)
					new_reward = transition.compute_reward(state, new_next_state, new_action)
					new_done = transition.not_healthy(new_next_state)

					# calculate the new a' for new next state
					noise = (
							torch.randn_like(action) * self.policy_noise
					).clamp(-self.noise_clip, self.noise_clip)

					new_next_action = self.actor_target(new_next_state, new_action) + noise
					new_next_action = self.beta * new_next_action + (1 - self.beta) * new_action
					new_next_action = new_next_action.clamp(-self.max_action, self.max_action)

					# Compute the target Q value
					new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action,
																	  new_action)

					new_target_Q = torch.min(new_target_Q1, new_target_Q2)
					new_target_Q = new_reward + self.discount * (1-new_done) * new_target_Q

					# filter with the error in two target Qs
					target_error = torch.abs((new_target_Q1 - new_target_Q2))
					max_error = torch.max(target_error)
					if self.save_info_for_debugging:
						self.variance_saver.append(max_error.item())
					if max_error > max_error_so_far:
						max_error_so_far = torch.clone(max_error)
					filter = torch.where(torch.rand_like(target_error) < target_error/max_error_so_far, 1, 0)
					new_target_Q *= filter

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, prev_action)
		if add_artificial_transitions:
			if add_artificial_transitions_type == 'MVE':
				current_Q1_H = []
				current_Q2_H = []
				for timestep in range(H+1):
					current_Q1, current_Q2 = self.critic(state_H[timestep], action_H[timestep], prev_action_H[timestep])
					current_Q1 *= filter_last_H[timestep]
					current_Q2 *= filter_last_H[timestep]
					current_Q1_H.append(current_Q1)
					current_Q2_H.append(current_Q2)

			elif add_artificial_transitions_type == 'ours':
					new_current_Q1, new_current_Q2 = self.critic(state, new_action, prev_action)
					new_current_Q1 *= filter
					new_current_Q2 *= filter

		# calculate critic loss
		if add_artificial_transitions:
			if add_artificial_transitions_type == 'MVE':
				critic_loss_list = []
				for timestep in range(H+1):
					critic_loss_list.append(F.mse_loss(current_Q1_H[timestep], target_Q_H[-timestep-1]) +\
							F.mse_loss(current_Q2_H[timestep], target_Q_H[-timestep-1]))
				critic_loss = sum(critic_loss_list)/(H+1)
			elif add_artificial_transitions_type == 'ours':
				critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) +\
						F.mse_loss(new_current_Q1, new_target_Q) + F.mse_loss(new_current_Q2, new_target_Q)
		else:
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		if self.save_info_for_debugging:
			self.critic_loss_saver.append(critic_loss.item())
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			if add_bc_loss:
				# sample from demonstrations
				demo_state, demo_action, demo_next_state, demo_reward, demo_prev_action, _ = \
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
			else:
				# Compute actor loss
				actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean()
			if self.save_info_for_debugging:
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

		if self.total_it % self.save_freq == 0 and self.save_info_for_debugging:
			np.save(f"./results/{self.file_name_critic}", self.critic_loss_saver)
			np.save(f"./results/{self.file_name_actor}", self.actor_loss_saver)
			np.save(f"./results/{self.file_name_variance}", self.variance_saver)


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
		
