import SCDM.TD3_plus_demos.TD3 as TD3
import dexterous_gym
import gym
import numpy as np
import time
from SCDM.TD3_plus_demos.main import env_statedict_to_state
from SCDM.TD3_plus_demos.invariance import TwoEggCatchUnderArmSymmetry

filename = "../TD3_plus_demos/models/TD3_EggCatchUnderarm-v0_0_EggCatchUnderarm-v0_with_normalizer"
policy_env_name = "EggCatchUnderarm-v0"
env_name = "TwoEggCatchUnderArm-v0"
beta = 0.7

policy_env = gym.make(policy_env_name)
steps = 75
# steps = 1000 #long run, "standard" episode is 250


class PolicyMigration():
    def __init__(self, kwargs, filename, env_name):
        self.policy = TD3.TD3(**kwargs)
        self.policy.load(filename)
        self.env = gym.make(env_name)
        self.timestep_throwing = 35

        self.two_egg_catch_sym = TwoEggCatchUnderArmSymmetry()

    def generate_action(self, state, prev_action, num_steps):
        new_action = np.zeros(self.env.action_space.shape[0])
        # For env EggCatchUnderarm-v0, egg is thrown from hand_2 to hand_1

        # hand_1
        # exchange state,action
        exchange_state = self.two_egg_catch_sym.symmetric_state(state)
        exchange_prev_action = self.two_egg_catch_sym.symmetric_action(prev_action)
        print(exchange_state[120:133])
        # print(exchange_state[146:153])
        state_1 = np.concatenate((exchange_state[0:120], exchange_state[120:133], exchange_state[146:153]))
        new_action_1 = self.policy.select_action(np.array(state_1), exchange_prev_action)
        # hand_2
        print(state[120:133])
        state_2 = np.concatenate((state[:120], state[120:133], state[146:153]))
        new_action_2 = self.policy.select_action(np.array(state_2), prev_action)
        if num_steps <= self.timestep_throwing:
            # two hand should execute the throwing policy
            # hand_1
            new_action[:20] = new_action_1[20:40]
            new_action[46:52] = new_action_1[40:46]
            # hand_2
            new_action[20:40] = new_action_2[20:40]
            new_action[40:46] = new_action_2[40:46]

        else:
            # two hand should execute the catching policy
            # hand_1
            new_action[0:20] = new_action_1[:20]
            new_action[46:52] = new_action_1[46:52]
            # hand_2
            new_action[20:40] = new_action_2[:20]
            new_action[40:46] = new_action_2[46:52]
        return new_action

    def eval_policy(self, seed=1, eval_episodes=1, render=True, delay=0.03):
        self.env.seed(seed + 100)

        avg_reward = 0
        for _ in range(eval_episodes):
            state_dict = self.env.reset()
            if render:
                self.env.render()
                time.sleep(delay)
            num_steps = 0
            prev_action = np.zeros((self.env.action_space.shape[0],))
            while num_steps < steps:
                state = env_statedict_to_state(state_dict, env_name)
                action = self.generate_action(state, prev_action, num_steps)
                state_dict, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
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


if __name__ == "__main__":
    kwargs = {
        "state_dim": env_statedict_to_state(policy_env.env._get_obs(), policy_env_name).shape[0],
        "action_dim": policy_env.action_space.shape[0],
        "beta": beta,
        "max_action": 1.0
    }
    PM = PolicyMigration(kwargs, filename, env_name)
    PM.eval_policy(seed=0, eval_episodes=1, render=True, delay=0.03)