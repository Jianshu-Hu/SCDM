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

		# invariant sample generator
		if env_name == 'TwoEggCatchUnderArm-v0':
			self.invariance_definition = invariance.TwoEggCatchUnderArmInvariance()
		elif env_name == 'EggCatchOverarm-v0':
			self.invariance_definition = invariance.CatchOverarmInvariance()
		elif env_name == 'EggCatchUnderarm-v0' or env_name == "EggCatchUnderarm-v1":
			self.invariance_definition = invariance.CatchUnderarmInvariance()
		else:
			print('Invariance is not implemented for these envs')

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


	def train(self, replay_buffer, demo_replay_buffer, invariance_replay_buffer_list, batch_size=100,
			  add_invariance=False, add_bc_loss=False):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, prev_action, ind = replay_buffer.sample(batch_size)
		state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
		next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = self.actor_target(next_state, action) + noise
			next_action = self.beta*next_action + (1-self.beta)*action
			next_action = (
				next_action
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, prev_action)

		if add_invariance:
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
		else:
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			if add_bc_loss:
				# sample from demonstrations
				demo_state, demo_action, demo_next_state, demo_reward, demo_prev_action, demo_ind = \
					demo_replay_buffer.sample(int(batch_size/10))
				demo_state = torch.FloatTensor(self.normaliser.normalize(demo_state.cpu().data.numpy())).to(device)
				demo_next_state = torch.FloatTensor(self.normaliser.normalize(demo_next_state.cpu().data.numpy())).to(device)
				policy_action = self.beta*self.actor(demo_state, demo_prev_action) + (1-self.beta)*demo_prev_action
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
				# Compute actor losse
				actor_loss = -self.critic.Q1(state, self.beta*self.actor(state, prev_action) + (1-self.beta)*prev_action, prev_action).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
		