import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class TransitionModel():
    def __init__(self, state_dim, action_dim, file_name, batch_size, env):
        self.forward_model = ForwardModel(state_dim, action_dim)
        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=1e-4)
        self.batch_size = batch_size

        self.normaliser = Normaliser(state_dim, default_clip_range=5.0)

        self.loss_saver = []
        self.save_freq = 10000
        self.total_iter = 0

        self.file_name_loss = file_name + "_transition_model_loss"

        self.env = env

        self.achieved_pose_index = -20
        self.goal_pose_index = -7

    def train(self, replay_buffer):
        self.total_iter += 1

        # Sample replay buffer
        state, action, next_state, reward, prev_action, all_invariance, ind = replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
        next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

        predicted_state = self.forward_model.forward(state, action)
        loss = F.mse_loss(predicted_state, next_state)

        self.loss_saver.append(loss.item())
        # Optimize the critic
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        if self.total_iter % self.save_freq == 0:
            np.save(f"./results/{self.file_name_loss}", self.loss_saver)

    def compute_reward(self, state):
        state = state.cpu().data.numpy()
        # Here the state is normalized while we need the state before normaliser to compute the reward
        inv_normalize_state = state*self.normaliser.std+self.normaliser.mean
        achieved_goal = inv_normalize_state[:, self.achieved_pose_index:self.achieved_pose_index+7]
        goal = inv_normalize_state[:, self.goal_pose_index:]
        reward = self.env.env.compute_reward(achieved_goal, goal, info=None)

        return torch.FloatTensor(reward.reshape([-1, 1])).to(device)


