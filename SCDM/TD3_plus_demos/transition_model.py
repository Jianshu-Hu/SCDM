import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym.envs.robotics.rotations as rotations

from SCDM.TD3_plus_demos.normaliser import Normaliser
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ForwardModel, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, state_dim)

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.l4(a)


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
    def __init__(self, state_dim, action_dim, file_name, env_name, batch_size, compute_reward, train_reward_model=False,
                 use_reward_wrapper=False):
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
        self.save_data = False
        self.total_iter = 0

        self.file_name_loss = file_name + "_transition_model_loss"
        self.file_name_reward_loss = file_name + "_reward_model_loss"

        self.embed_compute_reward = compute_reward
        self.train_reward_model = train_reward_model
        self.use_reward_wrapper = use_reward_wrapper

        env_list1 = ['EggCatchOverarm-v0', 'EggCatchUnderarm-v0', 'EggCatchUnderarmHard-v0']
        env_list2 = ['PenSpin-v0']
        env_list3 = ['Reacher-v2']
        env_list4 = ['Pusher-v2']
        env_list5 = ['HalfCheetah-v3', 'Walker2d-v3', 'Swimmer-v3', 'Hopper-v3','Ant-v3']
        env_list6 = ['FetchSlideDense-v1', 'FetchPickAndPlaceDense-v1', 'FetchPushDense-v1',
                     'FetchSlideSparse-v1', 'FetchPickAndPlaceSparse-v1', 'FetchPushSparse-v1']
        self.env_name = env_name
        if env_name in env_list1:
            self.reward_type ='Egg'
            self.achieved_pose_index = -20
            self.goal_pose_index = -7
        elif env_name in env_list2:
            self.reward_type = 'PenSpin-v0'
            self.qpos_index = -7
            self.qvel_index = -13
            # These two are parameters in the PenSpin-v0 Environment.
            self.direction = 1
            self.alpha = 1.0
        elif env_name in env_list3:
            self.reward_type = 'Reacher-v2'
        elif env_name in env_list4:
            self.reward_type = "Pusher-v2"
        elif env_name in env_list5:
            self.reward_type = "Mujoco_robot"
            if env_name == 'HalfCheetah-v3':
                self.forward_reward_weight = 1.0
                self.ctrl_cost_weight = 0.1
                self.healthy_reward = 0
                self.x_vel_index = 8

                if self.use_reward_wrapper:
                    self.forward_reward_weight = 0.6
            elif env_name == 'Walker2d-v3':
                self.forward_reward_weight = 1.0
                self.ctrl_cost_weight = 0.001
                self.healthy_reward = 1.0
                self.x_vel_index = 8

                self.health_z_range = np.array([0.8, 2])
                self.health_angle_range = np.array([-1.0, 1.0])
            elif env_name =='Swimmer-v3':
                self.forward_reward_weight = 1.0
                self.ctrl_cost_weight = 0.001
                self.healthy_reward = 0
                self.x_vel_index = 3
            elif env_name =='Hopper-v3':
                self.forward_reward_weight = 1.0
                self.ctrl_cost_weight = 0.001
                self.healthy_reward = 1.0
                self.x_vel_index = 5

                self.health_z_range = np.array([0.7, np.inf])
                self.health_angle_range = np.array([-0.2, 0.2])
                self.health_state_range = np.array([-100.0, 100.0])
            elif env_name == 'Ant-v3':
                self.forward_reward_weight = 1.0
                self.ctrl_cost_weight = 0.5
                self.contact_cost_weight = 0.0005
                self.healthy_reward = 1.0
                self.x_vel_index = 13
                self.contact_index = 27

                self.health_angle_range = np.array([0.2, 1])
                self.contact_force_range = np.array([-1, 1])
        elif env_name in env_list6:
                self.reward_type = 'Fetch'
                self.achieved_goal_index = 3
                self.goal_index = -3
        else:
            raise NotImplementedError('Check the reward function for this environment')

    def train(self, replay_buffer):
        self.total_iter += 1

        # Sample replay buffer
        state, action, next_state, reward, prev_action, _, = replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(self.normaliser.normalize(state.cpu().data.numpy())).to(device)
        next_state = torch.FloatTensor(self.normaliser.normalize(next_state.cpu().data.numpy())).to(device)

        predicted_state = self.forward_model.forward(state, action)
        loss = F.mse_loss(predicted_state, next_state)

        if self.save_data:
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

            if self.save_data:
                self.reward_loss_saver.append(loss_reward.item())

            self.reward_optimizer.zero_grad()
            loss_reward.backward()
            self.reward_optimizer.step()

            if self.total_iter % self.save_freq == 0:
                np.save(f"./results/{self.file_name_reward_loss}", self.reward_loss_saver)


    # true reward
    def compute_reward(self, prev_state, state, action):
        prev_state = prev_state.cpu().data.numpy()
        state = state.cpu().data.numpy()
        action = action.cpu().data.numpy()
        # Here the state is normalized while we need the state before normaliser to compute the reward
        state = state * self.normaliser.std + self.normaliser.mean
        prev_state = prev_state * self.normaliser.std + self.normaliser.mean
        if self.reward_type == "PenSpin-v0":
            obj_qpos = state[:, self.qpos_index:]
            obj_qvel = state[:, self.qvel_index:self.qvel_index+6]

            rotmat = rotations.quat2mat(obj_qpos[:, -4:])
            bot = (rotmat @ np.array([[0], [0], [-0.1]])).reshape(state.shape[0], 3)
            top = (rotmat @ np.array([[0], [0], [0.1]])).reshape(state.shape[0], 3)
            reward_1 = -15 * np.abs(bot[:, -1] - top[:, -1])
            reward_2 = self.direction*obj_qvel[:, 3]
            reward = self.alpha*reward_2 + reward_1
        elif self.reward_type == "Reacher-v2":
            reward_1 = -np.linalg.norm(prev_state[:, -3:], axis=1)
            reward_2 = -np.sum(np.square(action), axis=1)
            reward = (reward_1 + reward_2).reshape(-1)
        elif self.reward_type == "Pusher-v2":
            vec1 = prev_state[:, -6:-3] - prev_state[:, -9:-6]
            vec2 = prev_state[:, -6:-3] - prev_state[:, -3:]
            reward_near = -np.linalg.norm(vec1, axis=1)
            reward_dist = -np.linalg.norm(vec2, axis=1)
            reward_ctrl = -np.sum(np.square(action), axis=1)
            reward = (reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near).reshape(-1)
        elif self.reward_type == "Mujoco_robot":
            if self.env_name == 'Ant-v3':
                forward_reward = self.forward_reward_weight * (prev_state[:, self.x_vel_index])
                ctrl_cost = -self.ctrl_cost_weight * np.sum(np.square(action), axis=1)
                contact_cost = -self.contact_cost_weight * np.sum(np.square(
                    np.clip(state[:, self.contact_index:], self. contact_force_range[0], self.contact_force_range[1])
                    ), axis=1)
                reward = (forward_reward + self.healthy_reward + ctrl_cost+contact_cost).reshape(-1)
            else:
                forward_reward = self.forward_reward_weight * (prev_state[:, self.x_vel_index])
                ctrl_cost = -self.ctrl_cost_weight * np.sum(np.square(action), axis=1)
                reward = (forward_reward + self.healthy_reward + ctrl_cost).reshape(-1)
        elif self.reward_type == "Fetch":
            achieved_goal = state[:, self.achieved_goal_index:self.achieved_goal_index+3]
            goal = state[:, self.goal_index:]
            reward = self.embed_compute_reward(achieved_goal, goal, info=None)
        else:
            achieved_goal = state[:, self.achieved_pose_index:self.achieved_pose_index+7]
            goal = state[:, self.goal_pose_index:]
            reward = self.embed_compute_reward(achieved_goal, goal, info=None)

        return torch.FloatTensor(reward.reshape([-1, 1])).to(device)

    def not_healthy(self, state):
        # for some envs, the agent might die and the epsiode will end

        state = state.cpu().data.numpy()
        # Here the state is normalized while we need the state before normaliser to compute the reward
        state = state * self.normaliser.std + self.normaliser.mean

        done = np.zeros(state.shape[0])

        if self.env_name == "Walker2d-v3":
            health_z = np.logical_and(self.health_z_range[0] < state[:, 0], state[:, 0] < self.health_z_range[1])
            health_angle = np.logical_and(self.health_angle_range[0] < state[:, 1], state[:, 1] < self.health_angle_range[1])
            done = 1-np.logical_and(health_z, health_angle)
        elif self.env_name == 'Hopper-v3':
            health_z = np.logical_and(self.health_z_range[0] < state[:, 0], state[:, 0] < self.health_z_range[1])
            health_angle = np.logical_and(self.health_angle_range[0] < state[:, 1], state[:, 1] < self.health_angle_range[1])
            health_state = np.all(np.logical_and(self.health_state_range[0] < state[:, 1:], state[:, 1:] < self.health_state_range[1]), axis=1)
            done = 1 - np.logical_and(np.logical_and(health_z, health_angle), health_state)
        elif self.env_name == 'Ant-v3':
            health_angle = np.logical_and(self.health_angle_range[0] <= state[:, 0], state[:, 0] <= self.health_angle_range[1])
            finite = np.all(np.isfinite(state), axis=1)
            done = 1-np.logical_and(finite, health_angle)
        return torch.FloatTensor(done.reshape([-1, 1])).to(device)

    def save(self, filename):
        torch.save(self.forward_model.state_dict(), filename + "_forward_model")
        torch.save(self.model_optimizer.state_dict(), filename + "_model_optimizer")

        joblib.dump(self.normaliser, filename + "_normaliser")

    def load(self, filename):
        self.forward_model.load_state_dict(torch.load(filename + "_forward_model"))
        self.model_optimizer.load_state_dict(torch.load(filename + "_model_optimizer"))

        self.normaliser = joblib.load(filename + "_normaliser")

