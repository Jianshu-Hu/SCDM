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
from SCDM.TD3_plus_demos.utils import env_statedict_to_state
import SCDM.TD3_plus_demos.transition_model as transition_model

# Runs policy for X episodes and returns average reward
# Runs critic for X episodes and returns average Q value for some states
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, target_rotation, eval_episodes=10, evaluate_critic_t=50):
	if env_name == 'PenSpin-v0':
		eval_env = gym.make(env_name)
	else:
		eval_env = gym.make(env_name, target_rotation=target_rotation)
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
	parser.add_argument("--demo_tag", type=str, default="")  #tag for the files of demonstration
	parser.add_argument("--add_bc_loss", action="store_true")  # add behavior cloning loss to actor training
	parser.add_argument("--model_start_timesteps", default=10000, type=int)# Time steps of start training transition model

	parser.add_argument("--add_invariance_traj", action="store_true")   # add invariant segment to the replay buffer
	parser.add_argument("--add_invariance_regularization", action="store_true")  # add regularization term to the loss of critic
	parser.add_argument("--add_hand_invariance_regularization",
						action="store_true")  # add regularization term to the loss of critic
	parser.add_argument("--add_artificial_transitions",
						action="store_true")  # add artificial transitions during the training
	parser.add_argument("--N_artificial_sample", type=int, default=1) #number of artificial samples generated
	parser.add_argument("--inv_type", type=str, default='translation')  # use translation or rotation
	parser.add_argument("--use_informative_segment", action="store_true") # use informative segment instead of restricted segment
	parser.add_argument("--demo_goal_type", type=str, default='Random') # set the goal of the segment from the demonstration
	# True: the true goal of the segment
	# Noisy: add noise to the true goal of the segment
	# Random: randomly sample a goal

	parser.add_argument("--target_rotation", type=str, default="xyz")  #set the target rotation
	# xyz: rotation in xyz
	# z: rotation in z
	# ignore: without rotation
	parser.add_argument("--sparse_reward", action="store_true")  # reward type

	parser.add_argument("--divide_demos_into_N_parts", type=int, default=1) #divide the demos into N parts
	parser.add_argument("--pd_throw_decay", type=float, default=0.99999)    # after each segment scale probability down by this amount

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

	if args.env == "PenSpin-v0":
		env_main = gym.make(args.env)
		env_demo = gym.make(args.env)
		env_reset = gym.make(args.env)
	elif args.sparse_reward:
		env_main = gym.make(args.env, target_rotation=args.target_rotation,
							reward_type="sparse", distance_threshold=0.2, rotation_threshold=0.2)
		env_demo = gym.make(args.env, target_rotation=args.target_rotation,
							reward_type="sparse", distance_threshold=0.2, rotation_threshold=0.2)
		env_reset = gym.make(args.env, target_rotation=args.target_rotation,
							reward_type="sparse", distance_threshold=0.2, rotation_threshold=0.2)
	else:
		env_main = gym.make(args.env, target_rotation=args.target_rotation)
		env_demo = gym.make(args.env, target_rotation=args.target_rotation)
		env_reset = gym.make(args.env, target_rotation=args.target_rotation)

	# demo_states = []
	# demo_prev_actions = []
	# demo_goals = []
	# files = os.listdir("demonstrations/demonstrations"+"_"+args.env+args.demo_tag)
	# files = [file for file in files if file.endswith(".pkl")]
	# for file in files:
	# 	traj = joblib.load("demonstrations/demonstrations" + "_" + args.env+args.demo_tag + "/" + file)
	# 	demo_states_traj = []
	# 	demo_prev_actions_traj = []
	# 	for k, state in enumerate(traj["sim_states"]):
	# 		demo_states_traj.append(state)
	# 		if k == 0:
	# 			prev_action = np.zeros((env_main.action_space.shape[0],))
	# 		else:
	# 			prev_action = traj["actions"][k-1]
	# 		demo_prev_actions_traj.append(prev_action.copy())
	# 	demo_states.append(demo_states_traj)
	# 	demo_prev_actions.append(demo_prev_actions_traj)
	# 	demo_goals.append(traj["goal"])
	demo_processor = utils.DemoProcessor(args.env, args.demo_tag, args.divide_demos_into_N_parts)
	demo_states_throw, demo_prev_actions_throw, demo_states_catch, demo_prev_actions_catch, demo_goals =\
		demo_processor.process()
	# if not dividing the demos, there will not be a decay for the throwing part.
	if args.divide_demos_into_N_parts == 1:
		args.pd_throw_decay = args.pd_decay

	# Set seeds
	np.random.seed(args.seed)
	env_main.seed(args.seed)
	env_main.action_space.np_random.seed(args.seed)
	env_demo.seed(args.seed+1)
	env_demo.action_space.np_random.seed(args.seed+1)
	env_reset.seed(args.seed+2)
	env_reset.action_space.np_random.seed(args.seed+2)
	torch.manual_seed(args.seed)
	
	state_dim = env_statedict_to_state(env_main.env._get_obs(), args.env).shape[0]
	action_dim = env_main.action_space.shape[0]
	max_action = float(env_main.action_space.high[0])

	kwargs = {
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
		"beta": args.beta
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	policy = TD3.TD3(**kwargs)
	transition = transition_model.TransitionModel(state_dim, action_dim, file_name, args.env, args.batch_size,
					env_main.env.compute_reward)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.env)
	demo_replay_buffer = utils.DemoReplayBuffer(state_dim, action_dim, args.env, args.demo_tag, env_demo)

	invariant_replay_buffer_list = []
	if args.add_invariance_traj:
		invariant_replay_buffer = utils.InvariantReplayBuffer(state_dim, action_dim, env_name=args.env,
															  max_size=args.segment_len)
	elif args.add_invariance_regularization:
		for i in range(args.N_artificial_sample):
			invariant_replay_buffer = utils.InvariantReplayBuffer(state_dim, action_dim, env_name=args.env,
																  max_size=replay_buffer.max_size)
			invariant_replay_buffer_list.append(invariant_replay_buffer)
	
	# Evaluate untrained policy
	avg_reward, avg_Q = eval_policy(policy, args.env, args.seed, target_rotation=args.target_rotation)
	evaluations_policy = [avg_reward]
	evaluations_critic = [avg_Q]

	total_timesteps = 0
	segment_timestep = 0
	main_episode_timesteps = 0
	main_episode_prev_ac = np.zeros((action_dim,))
	main_episode_obs = env_main.reset()

	pd_prob = args.pd_init
	pd_throw_prob = args.pd_init
	pr_prob = args.pr_init

	print("-----start iteration with parameters-----")
	# print information
	print("seed: ", args.seed)
	print("env_name: ", args.env)
	print("demo_goal_type: ", args.demo_goal_type)
	print("add_behavior_cloning_loss: ", args.add_bc_loss)
	print("add_invariance_traj: ", args.add_invariance_traj)
	print("add_invariance_regularization: ", args.add_invariance_regularization)
	print("add_hand_invariance_regularization: ", args.add_hand_invariance_regularization)
	if args.add_invariance_traj or args.add_invariance_regularization or \
			args.add_hand_invariance_regularization:
		print("inv_type: ", args.inv_type)
		print("use_informative_segment: ", args.use_informative_segment)
	if args.sparse_reward:
		print("reward type: sparse reward")
	else:
		print("reward type: dense reward")
	print("target rotation: ", args.target_rotation)

	for t in range(int(args.max_timesteps)):
		
		if segment_timestep % args.segment_len == 0:
			segment_timestep = 0
			if t > 0:
				if args.pd_throw_decay > args.pd_decay:
					raise ValueError("The pd_throw should decay faster than pd")
				pd_prob *= args.pd_decay
				pd_throw_prob *= args.pd_throw_decay
				pr_prob *= args.pr_decay
			rn = np.random.rand()
			if rn < pd_prob:
				segment_type = "pd"
				if rn < pd_throw_prob:
					demo_type = "throw"
				else:
					demo_type = "catch"
			elif rn < pd_prob + pr_prob:
				segment_type = "pr"
			else:
				segment_type = "full"
			if segment_type == "pd":
				if demo_type == "throw":
					traj_ind = np.random.randint(0, len(demo_states_throw))
					state_ind = np.random.randint(0, len(demo_states_throw[traj_ind]))
					env_demo.reset()
					env_demo.env.sim.set_state(demo_states_throw[traj_ind][state_ind])
					prev_action = demo_prev_actions_throw[traj_ind][state_ind]
				else:
					traj_ind = np.random.randint(0, len(demo_states_catch))
					state_ind = np.random.randint(0, len(demo_states_catch[traj_ind]))
					env_demo.reset()
					env_demo.env.sim.set_state(demo_states_catch[traj_ind][state_ind])
					prev_action = demo_prev_actions_catch[traj_ind][state_ind]
				if args.env == 'PenSpin-v0':
					env_demo.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
					env_demo.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
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
				observation_dict = main_episode_obs
				observation = env_statedict_to_state(observation_dict, env_name=args.env)
		if t < args.start_timesteps:
			action = (
				policy.beta*env_main.action_space.sample() + (1-policy.beta)*prev_action
			).clip(-max_action, max_action)
		else:
			noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			action = (
				policy.select_action(observation, prev_action, noise=noise)).clip(-max_action, max_action)

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
		# replay_buffer.add(observation, action, next_observation, reward, prev_action)
		if args.add_invariance_traj:
			replay_buffer.add(observation, action, next_observation, reward, prev_action)
			invariant_replay_buffer.add(observation, action, next_observation, reward, prev_action)
			if (segment_timestep % args.segment_len == 0) and (segment_timestep > 0):
				invariant_replay_buffer.create_invariant_trajectory(inv_type=args.inv_type,
					use_informative=args.use_informative_segment, policy=policy)
				replay_buffer.add_from_other_replay_buffer(invariant_replay_buffer.state,
									invariant_replay_buffer.action, invariant_replay_buffer.next_state,
									invariant_replay_buffer.reward, invariant_replay_buffer.prev_action)
		else:
			replay_buffer.add(observation, action, next_observation, reward, prev_action)

		if args.add_invariance_regularization:
			for i in range(args.N_artificial_sample):
				invariant_replay_buffer_list[i].add_inv_sample(observation, action, next_observation, reward,
															   prev_action, args.inv_type)

		if args.use_normaliser:
			policy.normaliser.update(observation)
			if t % args.update_normaliser_every == 0:
				policy.normaliser.recompute_stats()

			transition.normaliser.update(observation)
			if t % args.update_normaliser_every == 0:
				transition.normaliser.recompute_stats()
		prev_action = action.copy()
		observation = next_observation.copy()
		if segment_type == "full":
			if main_episode_timesteps == env_main._max_episode_steps:
				main_episode_timesteps = 0
				prev_action = main_episode_prev_ac = np.zeros((action_dim,))
				observation = main_episode_obs = env_main.reset()

		if t >= args.model_start_timesteps:
			transition.train(replay_buffer)
		if t >= args.start_timesteps:
			policy.train(replay_buffer, demo_replay_buffer, invariant_replay_buffer_list, transition,
						 args.batch_size,
						 args.add_invariance_regularization, args.add_hand_invariance_regularization,
						 args.add_bc_loss,
						 args.add_artificial_transitions)

		if (t+1) % args.eval_freq == 0:
			avg_reward, avg_Q = eval_policy(policy, args.env, args.seed, target_rotation=args.target_rotation)
			evaluations_policy.append(avg_reward)
			evaluations_critic.append(avg_Q)
			np.save(f"./results/{file_name}", evaluations_policy)
			np.save(f"./results_critic/{file_name}", evaluations_critic)
			if args.save_model: policy.save(f"./models/{file_name}")
			print("Evaluation after %d steps - average reward: %f" % (total_timesteps, evaluations_policy[-1]))
			print("Evaluation after %d steps - average Q: %f" % (total_timesteps, evaluations_critic[-1]))
