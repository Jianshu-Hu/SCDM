import SCDM.TD3_plus_demos.TD3 as TD3
import dexterous_gym
import gym
import numpy as np
import time
from main import env_statedict_to_state

# filename = "models/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_her"
# env_name = "EggCatchOverarm-v0"
# filename = "models/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_her_exclude_demo"
# env_name = "EggCatchOverarm-v0"
# filename = "models/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_normalizer"
# env_name = "EggCatchOverarm-v0"
filename = "models/TD3_EggCatchOverarm-v0_0_1_random_goal_demo_with_01_translation"
env_name = "EggCatchOverarm-v1"
# filename = "models/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_restricted_translation_traj"
# env_name = "EggCatchOverarm-v0"
# filename = "models/TD3_EggCatchOverarm-v0_0_EggCatchOverarm-v0_with_rotation_traj"
# env_name = "EggCatchOverarm-v0"

# filename = "models/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_her"
# env_name = "EggCatchUnderarm-v0"
# filename = "models/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_her_exclude_demo"
# env_name = "EggCatchUnderarm-v0"
# filename = "models/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_translation_traj"
# env_name = "EggCatchUnderarm-v0"

# filename = "models/TD3_TwoEggCatchUnderArm-v0_0_without_symmetry_traj_without_normalizer"
# env_name = "TwoEggCatchUnderArm-v0"
# filename = "models/TD3_TwoEggCatchUnderArm-v0_0_add_symmetry_to_replay_buffer"
# env_name = "TwoEggCatchUnderArm-v0"

# filename = "models/TD3_EggHandOver-v0_0_EggHandOver-v0_with_normalizer"
# env_name = "EggHandOver-v0"

# filename = "models/TD3_BlockHandOver-v0_0_BlockHandOver-v0_with_normalizer"
# env_name = "BlockHandOver-v0"

# filename = "models/TD3_PenSpin-v0_0_PenSpin"
# env_name = "PenSpin-v0"

beta = 0.7

env = gym.make(env_name)
steps = 75
# steps = 1000 #long run, "standard" episode is 250


def eval_policy(policy, env_name, seed, eval_episodes=1, render=True, delay=0.0):
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
            if render:
                eval_env.render()
                time.sleep(delay)
            prev_action = action.copy()
            avg_reward += reward
            num_steps += 1
            print(num_steps)

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
    "max_action": 1.0
}

policy = TD3.TD3(**kwargs)
policy.load(filename)

eval_policy(policy, env_name, seed=1, eval_episodes=5, render=True, delay=0.03)