#built directly on top of author's TD3 implementation: https://github.com/sfujim/TD3

import numpy as np
import torch
import gym
import argparse
import os
import dexterous_gym
import joblib

import SCDM.TD3_plus_demos.utils as utils
import SCDM.TD3_plus_demos.TD3 as TD3
import SCDM.TD3_plus_demos.SAC as SAC
from SCDM.TD3_plus_demos.utils import env_statedict_to_state
import SCDM.TD3_plus_demos.transition_model as transition_model

import SCDM.TD3_plus_demos.wrapper as wrapper

# Runs policy for X episodes and returns average reward
# Runs critic for X episodes and returns average Q value for some states
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, use_wrapper=False, ignore_ori=False, use_reward_wrapper=False,
				eval_episodes=10, evaluate_critic_t=1):
	if use_wrapper:
		eval_env = wrapper.BasicWrapper(gym.make(env_name))
	elif ignore_ori:
		eval_env = gym.make(env_name,target_rotation='ignore')
	elif use_reward_wrapper:
		eval_env = wrapper.CheetahRewardWrapper(gym.make(env_name))
	else:
		eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	avg_Q = 0.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	for _ in range(eval_episodes):
		state_dict = eval_env.reset()
		num_steps = 0
		prev_action = np.zeros((eval_env.action_space.shape[0],))
		while num_steps < eval_env._max_episode_steps:
			state = env_statedict_to_state(state_dict, env_name)
			action = policy.select_action(state, prev_action)
			state_dict, reward, done, _ = eval_env.step(action)
			prev_action = action.copy()
			avg_reward += reward
			num_steps += 1

			if num_steps==evaluate_critic_t:
				state_critic = env_statedict_to_state(state_dict, env_name)
				action_critic = policy.select_action(state_critic, prev_action)

				state_critic = torch.FloatTensor(policy.normaliser.normalize(state_critic)).to(device)
				state_critic = torch.reshape(state_critic, (1, -1))
				prev_action_critic = torch.FloatTensor(prev_action.reshape(1, -1)).to(device)
				action_critic = torch.FloatTensor(action_critic.reshape(1, -1)).to(device)

				Q1, Q2 = policy.critic(state_critic, action_critic, prev_action_critic)
				avg_Q += (Q1.item() + Q2.item()) / 2

			if done:
				break

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")

	avg_Q /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation of Q over {eval_episodes} episodes: {avg_Q:.3f}")
	print("---------------------------------------")

	return avg_reward, avg_Q


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG, SAC)
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
	parser.add_argument("--load_dynamics_model", default="")  		# Model load file name, "" doesn't load, "default" uses file_name

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
	parser.add_argument("--without_demo", action='store_true')  # tag for not using demonstration
	parser.add_argument("--demo_tag", type=str, default="")  #tag for the files of demonstration
	parser.add_argument("--add_bc_loss", action="store_true")  # add behavior cloning loss to actor training
	parser.add_argument("--model_start_timesteps", default=10000, type=int)# Time steps of start training transition model

	parser.add_argument("--add_artificial_transitions_type", type=str, default='None')  # add artificial transitions during the training
	# None: without artificial transitions
	# ours: Using decaying noisy actions
	# MVE: Using Model-based Value Expansion
	# MA: Only use the filter with fixed action noise
	parser.add_argument("--prediction_horizon", type=int, default=1)  #the prediction horizon for MVE
	parser.add_argument("--demo_goal_type", type=str, default='Random') # set the goal of the segment from the demonstration
	# True: the true goal of the segment
	# Noisy: add noise to the true goal of the segment
	# Random: randomly sample a goal

	parser.add_argument("--critic_freq", type=int, default=1)  # increase the frequency of updating the critics

	# use wrapper to change the environment
	parser.add_argument("--use_wrapper", action='store_true')  # tag for using the wrapper
	# fix the target orientation
	parser.add_argument("--ignore_ori", action='store_true')  # tag for fixing target orientation of the object
	# use wrapper for testing generalization
	parser.add_argument("--use_reward_wrapper", action='store_true')  # tag for using the wrapper


	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_{args.expt_tag}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if not os.path.exists("./results_critic"):
		os.makedirs("./results_critic")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.use_wrapper:
		env_main = wrapper.BasicWrapper(gym.make(args.env))
		env_demo = wrapper.BasicWrapper(gym.make(args.env))
		env_reset = wrapper.BasicWrapper(gym.make(args.env))
	elif args.ignore_ori:
		env_main = gym.make(args.env, target_rotation='ignore')
		env_demo = gym.make(args.env, target_rotation='ignore')
		env_reset = gym.make(args.env, target_rotation='ignore')
	elif args.use_reward_wrapper:
		env_main = wrapper.CheetahRewardWrapper(gym.make(args.env))
		env_demo = wrapper.CheetahRewardWrapper(gym.make(args.env))
		env_reset = wrapper.CheetahRewardWrapper(gym.make(args.env))
	else:
		env_main = gym.make(args.env)
		env_demo = gym.make(args.env)
		env_reset = gym.make(args.env)


	if args.without_demo:
		args.pd_init = 0
		args.pr_init = 0
	else:
		demo_processor = utils.DemoProcessor(args.env, args.demo_tag)
		demo_states, demo_prev_actions, demo_goals = demo_processor.process()

	# Set seeds
	np.random.seed(args.seed)
	env_main.seed(args.seed)
	env_main.action_space.np_random.seed(args.seed)
	env_demo.seed(args.seed+1)
	env_demo.action_space.np_random.seed(args.seed+1)
	env_reset.seed(args.seed+2)
	env_reset.action_space.np_random.seed(args.seed+2)
	torch.manual_seed(args.seed)

	if args.use_wrapper:
		state_dim = env_statedict_to_state(env_main._get_obs(), args.env).shape[0]
	elif args.use_reward_wrapper:
		state_dim = env_statedict_to_state(env_main.env.env._get_obs(), args.env).shape[0]
	else:
		state_dim = env_statedict_to_state(env_main.env._get_obs(), args.env).shape[0]
	action_dim = env_main.action_space.shape[0]
	max_action = float(env_main.action_space.high[0])


	kwargs = {
		"policy_type": args.policy,
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"env_name": args.env,
		"file_name": file_name,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"beta": args.beta,
		"add_artificial_transitions_type":args.add_artificial_transitions_type
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	if args.policy == 'SAC':
		policy = SAC.SAC(**kwargs)
	else:
		policy = TD3.TD3(**kwargs)
	if hasattr(env_main.env, 'compute_reward'):
		transition = transition_model.TransitionModel(state_dim, action_dim, file_name, args.env, args.batch_size,
						env_main.env.compute_reward)
	elif args.use_reward_wrapper:
		transition = transition_model.TransitionModel(state_dim, action_dim, file_name, args.env, args.batch_size,
													  None, args.use_reward_wrapper)
	else:
		# there is not a reward function in the some envs
		transition = transition_model.TransitionModel(state_dim, action_dim, file_name, args.env, args.batch_size,
													  None)
	# initialize the max reward for setting the bias before start training the policy
	max_reward = -10000

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	if args.load_dynamics_model != "":
		dynamics_model_file = file_name if args.load_dynamics_model == "default" else args.load_dynamics_model
		transition.load(f"./models/{dynamics_model_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.env)
	if not args.without_demo:
		demo_replay_buffer = utils.DemoReplayBuffer(state_dim, action_dim, args.env, args.demo_tag, env_demo)

	
	# Evaluate untrained policy
	avg_reward, avg_Q = eval_policy(policy, args.env, args.seed, args.use_wrapper, args.ignore_ori,args.use_reward_wrapper)
	evaluations_policy = [avg_reward]
	evaluations_critic = [avg_Q]

	total_timesteps = 0
	segment_timestep = 0
	main_episode_timesteps = 0
	main_episode_prev_ac = np.zeros((action_dim,))
	main_episode_obs = env_statedict_to_state(env_main.reset(), env_name=args.env)

	pd_prob = args.pd_init
	pr_prob = args.pr_init

	print("-----start iteration with parameters-----")
	# print information
	print("seed: ", args.seed)
	print("env_name: ", args.env)
	print("demo_goal_type: ", args.demo_goal_type)
	print("add_behavior_cloning_loss: ", args.add_bc_loss)
	print("add_artificial_transitions: ", args.add_artificial_transitions_type)
	print("without_demo: ", args.without_demo)

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
				traj_ind = np.random.randint(0, len(demo_states))
				state_ind = np.random.randint(0, len(demo_states[traj_ind]))
				env_demo.reset()
				env_demo.env.sim.set_state(demo_states[traj_ind][state_ind])
				prev_action = demo_prev_actions[traj_ind][state_ind]
				if args.env == 'PenSpin-v0':
					env_demo.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
					env_demo.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
				if args.env == 'Reacher-v2' or 'Hopper-v3' or 'Walker2d-v3':
					pass
				else:
					# set the goal of the segment from the demonstrations
					if args.demo_goal_type == "True":
						env_demo.goal = np.copy(demo_goals[traj_ind])
						env_demo.env.goal = np.copy(demo_goals[traj_ind])
					elif args.demo_goal_type == "Noisy":
						random_goal = env_demo.env._sample_goal()
						true_goal = np.copy(demo_goals[traj_ind])
						env_demo.goal = 0.9*true_goal+random_goal*0.1
						env_demo.env.goal = 0.9*true_goal+random_goal*0.1
					elif args.demo_goal_type == "Random":
						random_goal = env_demo.env._sample_goal()
						env_demo.goal = np.copy(random_goal)
						env_demo.env.goal = np.copy(random_goal)
					else:
						raise ValueError("Wrong type for the goal of the demostration")
				pre_obs_dict = env_demo.env._get_obs()
				env_demo.env.sim.forward()
				observation_dict = env_demo.env._get_obs()
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
			elif segment_type == "pr":
				observation_dict = env_reset.reset()
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
				prev_action = np.zeros((action_dim,))
			else:
				prev_action = main_episode_prev_ac
				observation = main_episode_obs
		if t < args.start_timesteps:
			action = (
				policy.beta*env_main.action_space.sample() + (1-policy.beta)*prev_action
			).clip(-max_action, max_action)
		else:
			if args.policy == 'SAC':
				action = policy.select_action(observation, prev_action, deterministic=False)
			else:
				noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				action = (
					policy.select_action(observation, prev_action, noise=noise)).clip(-max_action, max_action)

		segment_timestep += 1
		total_timesteps += 1

		if segment_type == "pd":
			next_observation_dict, reward, done, _ = env_demo.step(action)
		elif segment_type == "pr":
			next_observation_dict, reward, done, _ = env_reset.step(action)
		else:
			main_episode_timesteps += 1
			next_observation_dict, reward, done, _ = env_main.step(action)
			# for robotics task keeps running, don't set the target Q value to zero for last time step
			if main_episode_timesteps == env_main._max_episode_steps:
				done = False
			main_episode_prev_ac = action.copy()
			main_episode_obs = env_statedict_to_state(next_observation_dict, env_name=args.env)

		next_observation = env_statedict_to_state(next_observation_dict, env_name=args.env)

		# replay buffer
		replay_buffer.add(observation, action, next_observation, reward, prev_action, done)
		if args.use_normaliser:
			policy.normaliser.update(observation)
			if t % args.update_normaliser_every == 0:
				policy.normaliser.recompute_stats()

			# if args.load_dynamics_model == "":
			transition.normaliser.update(observation)
			if t % args.update_normaliser_every == 0:
				transition.normaliser.recompute_stats()

		# set the prev_action and obs
		prev_action = action.copy()
		observation = next_observation.copy()
		if segment_type == "full":
			if (main_episode_timesteps == env_main._max_episode_steps) or done:
				main_episode_timesteps = 0
				prev_action = main_episode_prev_ac = np.zeros((action_dim,))
				observation_dict = env_main.reset()
				observation = main_episode_obs = env_statedict_to_state(observation_dict, env_name=args.env)

		# set the bias
		if reward > max_reward:
			max_reward = reward
		if args.add_artificial_transitions_type == 'ours':
			if (t + 1) % args.eval_freq == 0 and total_timesteps < args.start_timesteps:
				print("max reward so far: ", max_reward)
			if total_timesteps == args.start_timesteps:
				print("return upper bound: ", max_reward*env_main._max_episode_steps)
				if args.env == 'EggCatchUnderarm-v0':
					if args.policy == 'SAC':
						bias = max_reward*env_main._max_episode_steps/3
					elif args.policy == 'TD3':
						bias = max_reward * env_main._max_episode_steps / 6
				elif args.env == 'EggCatchOverarm-v0':
					if args.policy == 'SAC':
						bias = max_reward*env_main._max_episode_steps/10
					elif args.policy == 'TD3':
						bias = max_reward * env_main._max_episode_steps / 6
				elif args.env == 'PenSpin-v0':
					if args.policy == 'SAC':
						bias = max_reward*env_main._max_episode_steps/15
					elif args.policy == 'TD3':
						bias = max_reward * env_main._max_episode_steps / 6
				elif args.env == 'HalfCheetah-v3':
					bias = max_reward * env_main._max_episode_steps / 30
				elif args.env == 'Swimmer-v3':
					if args.policy == 'SAC':
						bias = max_reward*env_main._max_episode_steps/60
					elif args.policy == 'TD3':
						bias = max_reward * env_main._max_episode_steps / 30
				else:
					bias = max_reward * env_main._max_episode_steps / 6
				torch.nn.init.constant_(policy.critic.l3.bias.data, bias)
				torch.nn.init.constant_(policy.critic.l6.bias.data, bias)
				torch.nn.init.constant_(policy.critic_target.l3.bias.data, bias)
				torch.nn.init.constant_(policy.critic_target.l6.bias.data, bias)

		if t >= args.model_start_timesteps:
			if args.add_artificial_transitions_type != 'None':
				transition.train(replay_buffer)
		if t >= args.start_timesteps:
			if args.without_demo:
				for _ in range(1, args.critic_freq):
					policy.train(replay_buffer, None, transition, args.batch_size, args.add_bc_loss,
								 args.add_artificial_transitions_type, args.prediction_horizon, True)
				policy.train(replay_buffer, None, transition, args.batch_size, args.add_bc_loss,
							 args.add_artificial_transitions_type, args.prediction_horizon)
			else:
				for _ in range(1, args.critic_freq):
					policy.train(replay_buffer, demo_replay_buffer, transition, args.batch_size, args.add_bc_loss,
								 args.add_artificial_transitions_type, args.prediction_horizon, True)
				policy.train(replay_buffer, demo_replay_buffer, transition, args.batch_size, args.add_bc_loss,
							 args.add_artificial_transitions_type, args.prediction_horizon)

		if (t+1) % args.eval_freq == 0:
			avg_reward, avg_Q = eval_policy(policy, args.env, args.seed, args.use_wrapper,
											args.ignore_ori, args.use_reward_wrapper)

			evaluations_policy.append(avg_reward)
			evaluations_critic.append(avg_Q)
			np.save(f"./results/{file_name}", evaluations_policy)
			np.save(f"./results_critic/{file_name}", evaluations_critic)

			if args.save_model:
				policy.save(f"./models/{file_name}")
				transition.save(f"./models/{file_name}")
			print("Evaluation after %d steps - average reward: %f" % (total_timesteps, evaluations_policy[-1]))
			print("Evaluation after %d steps - average Q: %f" % (total_timesteps, evaluations_critic[-1]))
