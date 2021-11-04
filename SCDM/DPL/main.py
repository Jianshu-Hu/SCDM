#built directly on top of author's TD3 implementation: https://github.com/sfujim/TD3

import numpy as np
import torch
import gym
import argparse
import os
import dexterous_gym
import joblib

import SCDM.DPL.utils as utils
import SCDM.DPL.TD3 as TD3
from SCDM.DPL.divider import OneObjCatchDivider, OneObjHandOverDivider, TwoObjDivider


# number of objects in the env
def obj_num(env_name):
	# one hand
	env_list1 = ['PenSpin-v0']
	# two hands with one object
	env_list2 = ['EggCatchOverarm-v0', 'EggCatchUnderarm-v0', 'EggHandOver-v0',
				 'BlockCatchOverarm-v0', 'BlockCatchUnderarm-v0', 'BlockHandOver-v0',
				 'PenCatchOverarm-v0', 'PenCatchUnderarm-v0', 'PenHandOver-v0']
	# two hands with two objects
	env_list3 = ['TwoEggCatchUnderArm-v0']
	if env_name in env_list1:
		num = 0
	elif env_name in env_list2:
		num = 1
	elif env_name in env_list3:
		num = 2
	else:
		raise ValueError('wrong env')
	return num


# return the concatenated state
def env_statedict_to_state(state_dict, env_name):
	num_obj = obj_num(env_name)
	if isinstance(state_dict, dict):
		if num_obj == 0:
			state = np.copy(state_dict["observation"])
		elif num_obj == 1:
			state = np.concatenate((state_dict["observation"],
									state_dict["desired_goal"]))
		elif num_obj == 2:
			state = np.concatenate((state_dict["observation"],
									state_dict["desired_goal"]['object_1'],
									state_dict["desired_goal"]['object_2']))
		else:
			raise ValueError('wrong env')
	elif isinstance(state_dict, np.ndarray):
		state = np.copy(state_dict)

	return state


def env_half_state_dim(env_name, state_dim):
	num_obj = obj_num(env_name)
	if num_obj == 1:
		half_state_dim = (state_dim-20)/2+20
	elif num_obj == 2:
		half_state_dim = (state_dim-40)/2+40
	else:
		raise ValueError('wrong env')

	return int(half_state_dim)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy_1, policy_2, divider, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state_dict = eval_env.reset()
		num_steps = 0
		prev_action = np.zeros((eval_env.action_space.shape[0],))
		prev_action_1, prev_action_2 = divider.one_hand_action(prev_action)
		while num_steps < eval_env._max_episode_steps:
			state = env_statedict_to_state(state_dict, env_name)
			state_1, state_2 = divider.one_hand_state(state)
			action_1 = policy_1.select_action(state_1, prev_action_1)
			action_2 = policy_1.select_action(state_2, prev_action_2)
			action = divider.two_hand_action(action_1, action_2)
			state_dict, reward, done, _ = eval_env.step(action)
			prev_action = action.copy()
			prev_action_1, prev_action_2 = divider.one_hand_action(prev_action)
			avg_reward += reward
			num_steps += 1

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="PenSpin-v0")              # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5000, type=int)      # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=10e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.98)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	#my parameters
	parser.add_argument("--beta", type=float, default=0.7)          # action coupling parameter
	parser.add_argument("--pd_init", type=float, default=0.7)       # initial probability of loading to demo for each segment
	parser.add_argument("--pr_init", type=float, default=0.2)       # initial probability of sampling segment from reset
	parser.add_argument("--pd_decay", type=float, default=0.999996)     # after each segment scale probability down by this amount
	parser.add_argument("--pr_decay", type=float, default=0.999996)     # after each segment scale probability down by this amount
	parser.add_argument("--segment_len", type=int, default=15)      # how long each "segment" is ran before resampling
	parser.add_argument("--use_normaliser", dest='use_normaliser', action='store_true') #use demos to normalise observations
	parser.add_argument("--update_normaliser_every", type=int, default=20)
	parser.add_argument("--expt_tag", type=str, default="")
	parser.set_defaults(use_normaliser=False)

	# new paramters
	parser.add_argument("--add_invariance", action="store_true")   # add invariant transition to the replay buffer
	parser.add_argument("--use_her", action="store_true")  # use hindsight replay buffer

	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_{args.expt_tag}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env_main = gym.make(args.env)
	env_demo = gym.make(args.env)
	env_reset = gym.make(args.env)

	demo_states = []
	demo_prev_actions = []
	files = os.listdir("demonstrations/demonstrations"+"_"+args.expt_tag)
	files = [file for file in files if file.endswith(".pkl")]
	for file in files:
		traj = joblib.load("demonstrations/demonstrations" + "_" + args.expt_tag + "/" + file)
		for k, state in enumerate(traj["sim_states"]):
			demo_states.append(state)
			if k==0:
				prev_action = np.zeros((env_main.action_space.shape[0],))
			else:
				prev_action = traj["actions"][k-1]
			demo_prev_actions.append(prev_action.copy())

	# Set seeds
	env_main.seed(args.seed)
	env_demo.seed(args.seed+1)
	env_reset.seed(args.seed+2)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)


	full_state_dim = env_statedict_to_state(env_main.env._get_obs(), args.env).shape[0]
	full_action_dim = env_main.action_space.shape[0]
	half_state_dim = env_half_state_dim(args.env, full_state_dim)
	half_action_dim = int(full_action_dim/2)
	max_action = float(env_main.action_space.high[0])

	# create divider for process state and action
	if obj_num(args.env) == 1:
		# the state and action spaces for HandOver and Catch are different
		if args.env == "EggHandOver-v0" or args.env == "BlockHandOver-v0" or args.env== "PenHandOver-v0":
			divider = OneObjHandOverDivider(half_state_dim, half_action_dim)
		else:
			divider = OneObjCatchDivider(half_state_dim, half_action_dim)
	elif obj_num(args.env) == 2:
		divider = TwoObjDivider(half_state_dim, half_action_dim)
	else:
		raise ValueError('Wrong number')

	# TODO:two policy can use two pairs of parameters
	kwargs = {
		"state_dim": half_state_dim,
		"action_dim": half_action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"beta": args.beta
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	policy_hand1 = TD3.TD3(**kwargs)
	policy_hand2 = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy_hand1.load(f"./models/{policy_file}" + "_hand1")
		policy_hand2.load(f"./models/{policy_file}" + "_hand2")

	# TODO:Instead of adding all the samples to the replay buffer, we might only add the samples which are related
	#  to one hand. For example, when the ball is in hand 1, the position of the obj has nothing to do with hand2 and
	#  this sample might not be added to the replay buffer of hand2.
	replay_buffer_1 = utils.ReplayBuffer(half_state_dim, half_action_dim, env_name=args.env, hand_idx=1)
	replay_buffer_2 = utils.ReplayBuffer(half_state_dim, half_action_dim, env_name=args.env, hand_idx=1)
	hindsight_replay_buffer_1 = utils.HindsightReplayBuffer(half_state_dim, half_action_dim, env_name=args.env,
										  segment_length=args.segment_len, compute_reward=env_main.env.compute_reward)
	hindsight_replay_buffer_2 = utils.HindsightReplayBuffer(half_state_dim, half_action_dim, env_name=args.env,
										  segment_length=args.segment_len, compute_reward=env_main.env.compute_reward)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy_hand1, policy_hand2, divider, args.env, args.seed)]

	total_timesteps = 0
	segment_timestep = 0
	main_episode_timesteps = 0
	main_episode_prev_ac = np.zeros((full_action_dim,))
	main_episode_obs = env_main.reset()

	pd_prob = args.pd_init
	pr_prob = args.pr_init

	for t in range(int(args.max_timesteps)):
		
		if segment_timestep % args.segment_len == 0:
			segment_timestep = 0
			if t > 0:
				pd_prob *= args.pd_decay
				pr_prob *= args.pr_decay
			rn = np.random.rand()
			if rn < pd_prob:
				segment_type = "pd"
			elif rn < pd_prob + pr_prob:
				segment_type = "pr"
			else:
				segment_type = "full"
			if segment_type == "pd":
				ind = np.random.randint(0, len(demo_states))
				env_demo.reset()
				env_demo.env.sim.set_state(demo_states[ind])
				pre_obs_dict = env_demo.env._get_obs()
				env_demo.env.sim.forward()
				observation_dict = env_demo.env._get_obs()
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
				prev_action = demo_prev_actions[ind]
			elif segment_type == "pr":
				observation_dict = env_reset.reset()
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
				prev_action = np.zeros((full_action_dim,))
			else:
				prev_action = main_episode_prev_ac
				observation_dict = main_episode_obs
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
		if t < args.start_timesteps:
			# policy hand1 and policy hand2 can use two different beta parameters
			# hand1
			full_action_1 = (
				policy_hand1.beta*env_main.action_space.sample() + (1-policy_hand1.beta)*prev_action
			).clip(-max_action, max_action)
			half_action_1, _ = divider.one_hand_action(full_action_1)
			# hand2
			full_action_2 = (
				policy_hand2.beta*env_main.action_space.sample() + (1-policy_hand2.beta)*prev_action
			).clip(-max_action, max_action)
			_, half_action_2 = divider.one_hand_action(full_action_2)
			# action
			action = divider.two_hand_action(half_action_1, half_action_2)

		else:
			# two hands can have different noise
			noise_1 = np.random.normal(0, max_action * args.expl_noise, size=half_action_dim)
			noise_2 = np.random.normal(0, max_action * args.expl_noise, size=half_action_dim)
			# forward the observation for two hands separately
			observation_hand1, observation_hand2 = divider.one_hand_state(observation)
			prev_action_hand1, prev_action_hand2 = divider.one_hand_action(prev_action)
			half_action_1 = (
				policy_hand1.select_action(observation_hand1, prev_action_hand1, noise=noise_1)
			).clip(-max_action, max_action)
			half_action_2 = (
				policy_hand2.select_action(observation_hand2, prev_action_hand2, noise=noise_2)
			).clip(-max_action, max_action)
			# get the full action
			action = divider.two_hand_action(half_action_1, half_action_2)


		segment_timestep += 1
		total_timesteps += 1

		if segment_type == "pd":
			next_observation_dict, reward, _, _ = env_demo.step(action)
		elif segment_type == "pr":
			next_observation_dict, reward, _, _ = env_reset.step(action)
		else:
			main_episode_timesteps += 1
			next_observation_dict, reward, _, _ = env_main.step(action)
			main_episode_prev_ac = action.copy()
			main_episode_obs = env_statedict_to_state(next_observation_dict, env_name=args.env)

		next_observation = env_statedict_to_state(next_observation_dict, env_name=args.env)

		# replay buffer
		# process the data
		observation_hand1, observation_hand2 = divider.one_hand_state(observation)
		next_observation_hand1, next_observation_hand2 = divider.one_hand_state(next_observation)
		action_hand1, action_hand2 = divider.one_hand_action(action)
		prev_action_hand1, prev_action_hand2 = divider.one_hand_action(prev_action)

		replay_buffer_1.add(observation_hand1, action_hand1, next_observation_hand1, reward, prev_action_hand1,
							args.add_invariance)
		replay_buffer_2.add(observation_hand2, action_hand2, next_observation_hand2, reward, prev_action_hand2,
							args.add_invariance)
		hindsight_replay_buffer_1.add(observation_hand1, action_hand1, next_observation_hand1, reward,
									  prev_action_hand1, add_invariance=False)
		hindsight_replay_buffer_2.add(observation_hand2, action_hand2, next_observation_hand2, reward,
									  prev_action_hand2, add_invariance=False)
		if args.use_her:
			if (segment_timestep % args.segment_len == 0) and (segment_timestep > 0):
				hindsight_replay_buffer_1.choose_new_goal()
				replay_buffer_1.add_from_hindsight_replay_buffer(hindsight_replay_buffer_1.state,
																 hindsight_replay_buffer_1.action,
																 hindsight_replay_buffer_1.next_state,
																 hindsight_replay_buffer_1.reward,
																 hindsight_replay_buffer_1.prev_action)

				hindsight_replay_buffer_2.choose_new_goal()
				replay_buffer_2.add_from_hindsight_replay_buffer(hindsight_replay_buffer_2.state,
																 hindsight_replay_buffer_2.action,
																 hindsight_replay_buffer_2.next_state,
																 hindsight_replay_buffer_2.reward,
																 hindsight_replay_buffer_2.prev_action)

		if args.use_normaliser:
			policy_hand1.normaliser.update(observation_hand1)
			if t % args.update_normaliser_every == 0:
				policy_hand1.normaliser.recompute_stats()
			policy_hand2.normaliser.update(observation_hand2)
			if t % args.update_normaliser_every == 0:
				policy_hand2.normaliser.recompute_stats()
		prev_action = action.copy()
		observation = next_observation.copy()
		if segment_type == "full":
			if main_episode_timesteps == env_main._max_episode_steps:
				main_episode_timesteps = 0
				prev_action = main_episode_prev_ac = np.zeros((full_action_dim,))
				observation = main_episode_obs = env_main.reset()

		if t >= args.start_timesteps:
			policy_hand1.train(replay_buffer_1, args.batch_size)
			policy_hand2.train(replay_buffer_2, args.batch_size)

		if (t+1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy_hand1, policy_hand2, divider, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model:
				policy_hand1.save(f"./models/{file_name}" + "_hand1")
				policy_hand2.save(f"./models/{file_name}" + "_hand2")
			print("Evaluation after %d steps - average reward: %f" % (total_timesteps, evaluations[-1]))