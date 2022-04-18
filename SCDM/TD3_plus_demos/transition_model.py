import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym.envs.robotics.rotations as rotations

from SCDM.TD3_plus_demos.normaliser import Normaliser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ForwardModel, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, state_dim)

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        return self.l3(a)


class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        return self.l3(a)


class TransitionModel():
    def __init__(self, state_dim, action_dim, file_name, env_name, batch_size, compute_reward, train_reward_model=False):
        self.forward_model = ForwardModel(state_dim, action_dim).to(device)
        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=1e-4)
        if train_reward_model:
            self.reward_model = RewardModel(state_dim, action_dim).to(device)
            self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        self.batch_size = batch_size

        self.normaliser = Normaliser(state_dim, default_clip_range=5.0)

        self.loss_saver = []
        self.reward_loss_saver = []
        self.save_freq = 10000
        self.total_iter = 0

        self.file_name_loss = file_name + "_transition_model_loss"
        self.file_name_reward_loss = file_name + "_reward_model_loss"

        self.embed_compute_reward = compute_reward
        self.train_reward_model = train_reward_model

        env_list1 = ['EggCatchOverarm-v0', 'EggCatchUnderarm-v0', 'EggCatchUnderarmHard-v0']
        env_list2 = ['PenSpin-v0']
        self.PenSpin_reward = False
        if env_name in env_list1:
            self.achieved_pose_index = -20
            self.goal_pose_index = -7
        elif env_name in env_list2:
            self.PenSpin_reward = True
            self.qpos_index = -7
            self.qvel_index = -13
            # These two are parameters in the PenSpin-v0 Environment.
            self.direction = 1
            self.alpha = 1.0
        else:
            print('Check the index for other environments')

    def train(self, replay_buffer):
        self.total_iter += 1

        # Sample replay buffer
        state, action, next_state, reward, prev_action, all_invariance, ind = replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
        next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

        predicted_state = self.forward_model.forward(state, action)
        loss = F.mse_loss(predicted_state, next_state)

        self.loss_saver.append(loss.item())

        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        if self.total_iter % self.save_freq == 0:
            np.save(f"./results/{self.file_name_loss}", self.loss_saver)

        if self.train_reward_model:
            predicted_reward = self.reward_model.forward(state, action)
            loss_reward = F.mse_loss(predicted_reward, reward)

            self.reward_loss_saver.append(loss_reward.item())

            self.reward_optimizer.zero_grad()
            loss_reward.backward()
            self.reward_optimizer.step()

            if self.total_iter % self.save_freq == 0:
                np.save(f"./results/{self.file_name_reward_loss}", self.reward_loss_saver)


    # true reward
    def compute_reward(self, state):
        state = state.cpu().data.numpy()
        if self.PenSpin_reward:
            obj_qpos = state[:, self.qpos_index:]
            obj_qvel = state[:, self.qvel_index:self.qvel_index+6]

            rotmat = rotations.quat2mat(obj_qpos[:, -4:])
            bot = (rotmat @ np.array([[0], [0], [-0.1]])).reshape(state.shape[0], 3)
            top = (rotmat @ np.array([[0], [0], [0.1]])).reshape(state.shape[0], 3)
            reward_1 = -15 * np.abs(bot[:, -1] - top[:, -1])
            reward_2 = self.direction*obj_qvel[:, 3]
            reward = self.alpha*reward_2 + reward_1
        else:
            # Here the state is normalized while we need the state before normaliser to compute the reward
            inv_normalize_state = state*self.normaliser.std+self.normaliser.mean
            achieved_goal = inv_normalize_state[:, self.achieved_pose_index:self.achieved_pose_index+7]
            goal = inv_normalize_state[:, self.goal_pose_index:]
            reward = self.embed_compute_reward(achieved_goal, goal, info=None)

        return torch.FloatTensor(reward.reshape([-1, 1])).to(device)


