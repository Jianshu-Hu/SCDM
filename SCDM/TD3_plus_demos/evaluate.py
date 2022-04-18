import SCDM.TD3_plus_demos.TD3 as TD3
import dexterous_gym
import gym
import numpy as np
import time
from main import env_statedict_to_state

# filename = "models/TD3_EggCatchOverarm-v0_0_random_goal_demo_2"
# env_name = "EggCatchOverarm-v0"
#
# filename = "models/TD3_EggCatchOverarm-v0_1_random_goal_demo_demo_divided_into_two_part_add_policy_penalty_all_actions"
# env_name = "EggCatchOverarm-v0"

# filename = "models/TD3_EggCatchUnderarm-v0_0_1_random_goal_demo_exclude_demo_egg_in_the_air"
# env_name = "EggCatchUnderarm-v0"

# filename = "models/TD3_EggHandOver-v0_0_EggHandOver-v0_with_normalizer"
# env_name = "EggHandOver-v0"

# filename = "models/TD3_BlockHandOver-v0_0_BlockHandOver-v0_with_normalizer"
# env_name = "BlockHandOver-v0"

filename = "models/TD3_PenSpin-v0_0_PenSpin"
env_name = "PenSpin-v0"

beta = 0.7

env = gym.make(env_name)
steps = 75
# steps = 1000 #long run, "standard" episode is 250


def compute_reward(state):
    obj_qpos = state[:, -7:]
    obj_qvel = state[:, -13:-7]

    rotmat = gym.envs.robotics.rotations.quat2mat(obj_qpos[:, -4:])
    bot = (rotmat @ np.array([[0], [0], [-0.1]])).reshape(state.shape[0], 3)
    top = (rotmat @ np.array([[0], [0], [0.1]])).reshape(state.shape[0], 3)
    reward_1 = -15 * np.abs(bot[:, -1]-top[:, -1])
    reward_2 = obj_qvel[:, 3]
    reward = reward_2 + reward_1
    return reward



def eval_policy(policy, env_name, seed, eval_episodes=1, render=True, delay=0.0):
    state_list = np.zeros([75, 61])

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state_dict = eval_env.reset()
        if render:
            eval_env.render()
            time.sleep(delay)
        num_steps = 0
        prev_action = np.zeros((eval_env.action_space.shape[0],))
        while num_steps < steps:
            state = env_statedict_to_state(state_dict, env_name)
            action = policy.select_action(np.array(state), prev_action)

            state_dict, reward, done, _ = eval_env.step(action)
            state_list[num_steps, :] = env_statedict_to_state(state_dict, env_name)
            if render:
                eval_env.render()
                time.sleep(delay)
            prev_action = action.copy()
            avg_reward += reward
            num_steps += 1
            print("num_steps: ", num_steps, "reward: ", reward)

    reward = compute_reward(state_list)
    print(reward)
    print(np.sum(reward))
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

kwargs = {
    "env_name": env_name,
    "state_dim": env_statedict_to_state(env.env._get_obs(), env_name).shape[0],
    "action_dim": env.action_space.shape[0],
    "beta": beta,
    "max_action": 1.0,
    "file_name": ""
}

policy = TD3.TD3(**kwargs)
policy.load(filename)

eval_policy(policy, env_name, seed=1, eval_episodes=1, render=False, delay=0.03)


