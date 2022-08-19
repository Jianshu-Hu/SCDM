import copy
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from SCDM.TD3_plus_demos.normaliser import Normaliser
from torch.distributions.normal import Normal
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Soft Actor Critic (SAC) from
# https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

	def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU, act_limit=1.0):
		super().__init__()
		self.net = mlp([obs_dim+act_dim] + list(hidden_sizes), activation, activation)
		self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
		self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
		self.act_limit = act_limit

	def forward(self, obs, prev_act, deterministic=False, with_logprob=True):
		net_out = self.net(torch.cat([obs, prev_act], 1))
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
		std = torch.exp(log_std)

		# Pre-squash distribution and sample
		pi_distribution = Normal(mu, std)
		if deterministic:
			# Only used for evaluating policy at test time.
			pi_action = mu
		else:
			pi_action = pi_distribution.rsample()

		if with_logprob:
			# Compute logprob from Gaussian, and then apply correction for Tanh squashing.
			# NOTE: The correction formula is a little bit magic. To get an understanding
			# of where it comes from, check out the original SAC paper (arXiv 1801.01290)
			# and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
			# Try deriving it yourself as a (very difficult) exercise. :)
			logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
			logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
			logp_pi.unsqueeze_(-1)
		else:
			logp_pi = None

		pi_action = torch.tanh(pi_action)
		pi_action = self.act_limit * pi_action

		return pi_action, logp_pi

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



class SAC(object):
	def __init__(
		self,
		policy_type,
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
		beta=0.7,
		initial_alpha=0.1,
		add_artificial_transitions_type='None'
	):
		# build policy and value functions
		self.actor = SquashedGaussianMLPActor(state_dim, action_dim, act_limit=max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		# if add_artificial_transitions_type == 'ours':
		# 	# initialize with a high Q value to encourage exploration
		# 	if env_name == 'PenSpin-v0':
		# 		nn.init.constant_(self.critic.l3.bias.data, 50)
		# 		nn.init.constant_(self.critic.l6.bias.data, 50)
		# 	elif env_name == 'EggCatchOverarm-v0':
		# 		nn.init.constant_(self.critic.l3.bias.data, 10)
		# 		nn.init.constant_(self.critic.l6.bias.data, 10)
		# 	elif env_name == 'EggCatchUnderarm-v0':
		# 		nn.init.constant_(self.critic.l3.bias.data, 10)
		# 		nn.init.constant_(self.critic.l6.bias.data, 10)
		# 	elif env_name == 'Walker2d-v3':
		# 		nn.init.constant_(self.critic.l3.bias.data, 100)
		# 		nn.init.constant_(self.critic.l6.bias.data, 100)
		# 	elif env_name == 'HalfCheetah-v3':
		# 		nn.init.constant_(self.critic.l3.bias.data, 50)
		# 		nn.init.constant_(self.critic.l6.bias.data, 50)
		# 	elif env_name == 'Swimmer-v3':
		# 		nn.init.constant_(self.critic.l3.bias.data, 30)
		# 		nn.init.constant_(self.critic.l6.bias.data, 30)
		# 	elif env_name == 'Hopper-v3':
		# 		nn.init.constant_(self.critic.l3.bias.data, 50)
		# 		nn.init.constant_(self.critic.l6.bias.data, 50)
		# 	elif env_name == 'Ant-v3':
		# 		nn.init.constant_(self.critic.l3.bias.data, 100)
		# 		nn.init.constant_(self.critic.l6.bias.data, 100)
		# 	elif env_name == 'Pusher-v2':
		# 		nn.init.constant_(self.critic.l3.bias.data, -10)
		# 		nn.init.constant_(self.critic.l6.bias.data, -10)
		# 	elif env_name == 'Reacher-v2':
		# 		nn.init.constant_(self.critic.l3.bias.data, -2)
		# 		nn.init.constant_(self.critic.l6.bias.data, -2)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau

		self.log_alpha = torch.tensor(np.log(initial_alpha)).to(device)
		self.log_alpha.requires_grad = True
		# set target entropy to -|A|
		self.target_entropy = -action_dim
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.beta = beta
		self.state_dim = state_dim

		self.policy_type = policy_type
		self.policy_freq = 1

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
		# self.epsilon_decay = 0.9999995

		self.noise_variance = 5
		self.noise_variance_decay = 0.9999995

		self.env_name = env_name

	def select_action(self, state, prev_action, deterministic=True):
		with torch.no_grad():
			state = torch.FloatTensor(self.normaliser.normalize(state.reshape(1, -1))).to(device)
			prev_action = torch.FloatTensor(prev_action.reshape(1, -1)).to(device)
			action, _ = self.actor(state, prev_action, deterministic, False)
		return self.beta*action.cpu().data.numpy().flatten() + \
			(1-self.beta)*prev_action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, demo_replay_buffer, transition, batch_size=100, add_bc_loss=False,
			  add_artificial_transitions_type=None, prediction_horizon=1, only_train_critic=False):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, prev_action, done, = replay_buffer.sample(batch_size)

		state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
		next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

		if add_artificial_transitions_type == "None":
			add_artificial_transitions = False
		else:
			add_artificial_transitions = True
			if add_artificial_transitions_type == 'ours':
				# gaussian, uniform, fixed
				noise_type = 'gaussian'
				initial_bound = self.max_action
				max_error_so_far = torch.zeros(1)
				use_the_filter = True
				if self.env_name == 'EggCatchUnderarm-v0' or self.env_name == 'EggCatchOverarm-v0':
					use_the_filter_of_higher_target_Q = False
				else:
					use_the_filter_of_higher_target_Q = True
			elif add_artificial_transitions_type == 'MA':
				noise_type = 'fixed'
				initial_bound = self.max_action
				max_error_so_far = torch.zeros(1)
				use_the_filter = True
				use_the_filter_of_higher_target_Q = False

		with torch.no_grad():
			# Select action according to policy
			next_action, logp_next_action = self.actor(next_state, action)
			next_action = self.beta * next_action + (1 - self.beta) * action
			next_action = (
				next_action
			).clamp(-self.max_action, self.max_action)

			# Target Q-values
			target_Q1, target_Q2 = self.critic_target(next_state, next_action, action)

			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * (1-done) * (target_Q-self.log_alpha.exp().detach()*logp_next_action)

			if add_artificial_transitions_type == 'ours' or add_artificial_transitions_type == 'MA':
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

				new_action, _ = self.actor(state, prev_action, deterministic=True, with_logprob=False)
				new_action += noise
				new_action = self.beta * new_action + (1 - self.beta) * prev_action
				new_action = new_action.clamp(-self.max_action, self.max_action)

				# forward with the action using the dynamic model
				new_next_state = transition.forward_model(state, new_action)

				new_reward = transition.compute_reward(state, new_next_state, new_action)
				new_done = transition.not_healthy(new_next_state)

				# calculate the new a' for new next state
				new_next_action, logp_new_next_action = self.actor(new_next_state, new_action)
				new_next_action = self.beta * new_next_action + (1 - self.beta) * new_action
				new_next_action = new_next_action.clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				new_target_Q1, new_target_Q2 = self.critic_target(new_next_state, new_next_action,
																  new_action)

				new_target_Q = torch.min(new_target_Q1, new_target_Q2)
				new_target_Q = new_reward + self.discount * (1-new_done) * \
							(new_target_Q-self.log_alpha.exp().detach()*logp_new_next_action)

				# filter with the error in two target Qs
				target_error = torch.abs((new_target_Q1 - new_target_Q2))
				max_error = torch.max(target_error)
				if self.save_info_for_debugging:
					self.variance_saver.append(max_error.item())
				if max_error > max_error_so_far:
					max_error_so_far = torch.clone(max_error)
				filter = torch.where(torch.rand_like(target_error) < target_error/max_error_so_far, 1, 0)

				# filter with target Q value
				filter2 = torch.where(new_target_Q > target_Q, 1, 0)
				if use_the_filter:
					new_target_Q *= filter
				if use_the_filter_of_higher_target_Q:
					new_target_Q *= filter2

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, prev_action)
		if add_artificial_transitions:
			if add_artificial_transitions_type == 'ours' or add_artificial_transitions_type == 'MA':
				new_current_Q1, new_current_Q2 = self.critic(state, new_action, prev_action)
				if use_the_filter:
					new_current_Q1 *= filter
					new_current_Q2 *= filter
				if use_the_filter_of_higher_target_Q:
					new_current_Q1 *= filter2
					new_current_Q2 *= filter2

		# calculate critic loss
		if add_artificial_transitions:
			if add_artificial_transitions_type == 'ours' or add_artificial_transitions_type == 'MA':
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
			# Compute actor loss
			pi, logp_pi = self.actor(state, prev_action)
			q1, q2 = self.critic(state, self.beta*pi + (1-self.beta)*pi, prev_action)
			q_pi = torch.min(q1, q2)

			# Entropy-regularized policy loss
			actor_loss = (self.log_alpha.exp().detach() * logp_pi - q_pi).mean()
			if self.save_info_for_debugging:
				self.actor_loss_saver.append(actor_loss.item())
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# optimize alpha
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.log_alpha.exp() *
						  (-logp_pi - self.target_entropy).detach()).mean()
			alpha_loss.backward()
			self.log_alpha_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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

		self.normaliser = joblib.load(filename+"_normaliser")
