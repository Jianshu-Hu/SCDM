from invariant_traj_generator import InvariantTrajGenerator
import joblib
from copy import deepcopy
import numpy as np
import argparse
import random
from scipy.spatial.transform import Rotation as R


# test applying transformation on the state and action
class CatchUnderarmTransformationTest(InvariantTrajGenerator):
    def __init__(self, args):
        super().__init__(args)
        # state
        self.hand1_mount_index = 0
        self.hand2_mount_index = 30
        self.obj_index = 60
        self.tar_index = 67
        # action
        self.hand1_mount_action_index = 46
        self.hand2_mount_action_index = 40

        # transformation
        self.tf_hand1_to_global = np.array([[-1, 0, 0],
                                      [0, 0, -1],
                                      [0, -1, 0]])
        self.tf_hand2_to_global = np.array([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, -1, 0]])

        self.tf_global_to_hand1 = np.linalg.inv(self.tf_hand1_to_global)
        self.tf_global_to_hand2 = np.linalg.inv(self.tf_hand2_to_global)

        # translation max bias
        self.x_max_bias = 0.01
        self.y_max_bias = 0.01
        self.z_max_bias = 0.01

        # rotation max bias
        self.zrot_max_bias = 0.01

        # rotation info
        self.central_point_in_global = np.array([1, 0.675, 0.35])
        self.hand1_mount_point_in_global = np.array([1, 1.35, 0.35])
        self.hand2_mount_point_in_global = np.array([1, 0, 0.35])

    def test_translation(self):
        print('---start testing---')
        traj_name = self.traj_prefix + self.files[0]
        traj = joblib.load(traj_name)
        traj_invariant = deepcopy(traj)

        # reflect success (success is the same)
        # reflect rewards (rewards are the same)

        # apply translation on the sim_states within a small range
        # x_max_bias = -0.06 #(hand_2 -0.06 to 0.33)
        # y_max_bias = -0.13 #(hand_2 -0.13 to 0.26)
        # z_max_bias = -0.03 #(hand_1 -0.03 to 0.12)
        # z_max_bias = 0.04  # (hand_2 -0.04 to 0.04)
        x_max_bias = -0.3 #(hand_2 -0.06 to 0.33)
        y_max_bias = 0.3 #(hand_2 -0.13 to 0.26)
        z_max_bias = 0.3 #(hand_1 -0.03 to 0.12)
        # hand_xyz_bias = (np.random.rand(3)-0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])
        hand_xyz_bias = np.array([x_max_bias, y_max_bias, z_max_bias])
        obj_xyz_bias = np.array([hand_xyz_bias[0], hand_xyz_bias[2], hand_xyz_bias[1]])
        for t in range(len(traj["sim_states"])):
            '''
            hand_1 (target in hand1)
            '''
            # hand_1_mount (pos)
            traj_invariant["sim_states"][t].qpos[0:3] = traj["sim_states"][t].qpos[:3] + hand_xyz_bias

            '''
            hand_2 (throwing from hand2)
            '''
            # hand_2_mount (pos)
            traj_invariant["sim_states"][t].qpos[30:33] = traj["sim_states"][t].qpos[30:33] + \
                                                          np.array([-1, 1, -1]) * hand_xyz_bias

            '''
            obj_1
            '''
            # obj_1 (pos)
            traj_invariant["sim_states"][t].qpos[60:63] = traj["sim_states"][t].qpos[60:63] + \
                                                          np.array([-1, -1, -1]) * obj_xyz_bias

            # obj_1_target (pos)
            traj_invariant["sim_states"][t].qpos[67:70] = traj["sim_states"][t].qpos[67:70] + \
                                                          np.array([-1, -1, -1]) * obj_xyz_bias

            traj["sim_states"][t].qpos[62] = -0.3
            traj_invariant["sim_states"][t].qpos[62] = -0.3
            traj["sim_states"][t].qpos[69] = -0.3
            traj_invariant["sim_states"][t].qpos[69] = -0.3

        # reflect best rewards_it
        # reflect goal
        traj_invariant["goal"][0:3] = traj["goal"][0:3] + np.array([-1, -1, -1]) * obj_xyz_bias

        for t in range(len(traj["actions"])):
            # hand_1_mount
            traj_invariant["actions"][t][46:49] = traj["actions"][t][46:49] + hand_xyz_bias*np.array([5, 0, 0])
        #     # hand_2_mount
        #     traj_invariant["actions"][t][40:43] = traj["actions"][t][40:43] + hand_xyz_bias*np.array([-5, 5, -12.5])

        timestep = 2
        for timestep in range(11, 12):
            self.env.reset()
            self.env.env.sim.set_state(traj["sim_states"][timestep])
            self.env.env.sim.forward()
            print('original position before action: ', self.env.env.sim.data.qpos[0:6])
            self.env.env.step(traj["actions"][timestep])
            print('original action: ', traj["actions"][timestep][46:49])
            original_pos = np.copy(self.env.env.sim.data.qpos[0:6])
            print('original position after action: ', original_pos)

            self.env.env.sim.set_state(traj_invariant["sim_states"][timestep])
            self.env.env.sim.forward()
            print('translated position before action: ', self.env.env.sim.data.qpos[0:6])
            self.env.env.step(traj_invariant["actions"][timestep])
            print('translated action: ', traj_invariant["actions"][timestep][46:49])
            translated_pos = np.copy(self.env.env.sim.data.qpos[0:6])
            print('translated position after action: ', translated_pos)
            print('difference: ', translated_pos-original_pos-np.array([x_max_bias, y_max_bias, z_max_bias, 0, 0, 0]))
            # if np.any(translated_pos-original_pos-np.array([x_max_bias, y_max_bias, z_max_bias, 0, 0, 0])>1e-5):
            #     print('difference: ', translated_pos-original_pos)
            #     print('wrong')
            #     break

        # print('------start to check the difference------')
        # min_error = 100
        # best_x = 0
        # for i in range(400, 600):
        #     self.env.reset()
        #     self.env.env.sim.set_state(traj_invariant["sim_states"][0])
        #     timestep = 0
        #     total_error = 0
        #     for t in range(len(traj["actions"])):
        #         # hand_1_mount
        #         traj_invariant["actions"][t][46:49] = traj["actions"][t][46:49] + hand_xyz_bias*np.array([5, 5, 12.5])
        #         # hand_2_mount
        #         traj_invariant["actions"][t][40:43] = traj["actions"][t][40:43] + hand_xyz_bias*np.array([-5, 5, -12.5])
        #     for action in traj_invariant["actions"]:
        #         self.env.env.sim.set_state(traj_invariant["sim_states"][timestep])
        #         self.env.step(traj_invariant["actions"][timestep])
        #         timestep = timestep + 1
        #         real_state = self.env.env.sim.data.qpos
        #         total_error += np.linalg.norm(real_state[30:33] - traj_invariant["sim_states"][timestep].qpos[30:33])
        #         # total_error = np.linalg.norm(real_state[30:33] - traj_invariant["sim_states"][timestep].qpos[30:33])
        #         print(np.linalg.norm(real_state[0:3] - traj_invariant["sim_states"][timestep].qpos[0:3]))
        #     if total_error < min_error:
        #         min_error = total_error
        #         best_x = i/100
        #     print('total_errors:', total_error, 'current_min_error: ', min_error, 'current_best_x: ', best_x)

    def test_rotation(self):
        print('---start testing---')
        traj_name = self.traj_prefix + self.files[3]
        traj = joblib.load(traj_name)
        traj_invariant = deepcopy(traj)

        # reflect success (success is the same)
        # reflect rewards (rewards are the same)

        # apply rotation on the sim_states within a small range
        # self.zrot_bias = (random.random() - 0.5) * self.zrot_max_bias
        # self.xyz_rot_bias = np.array([0, 0, self.zrot_bias])
        # self.bias_r = R.from_rotvec(self.xyz_rot_bias)
        self.set_rotation_bias(traj["actions"])

        for t in range(len(traj["sim_states"])):
            traj_invariant["sim_states"][t].qpos[:] = self.rotation_inv_state(traj["sim_states"][t].qpos[:])

        # reflect goal
        obj_pos_in_central = traj["goal"][0:3] - self.central_point_in_global
        traj_invariant["goal"][0:3] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                          .reshape(-1) + self.central_point_in_global

        original_r = R.from_quat(traj["goal"][3:7])
        traj_invariant["goal"][3:7] = (self.bias_r * original_r).as_quat()
        if self.env_name == "EggCatchOverarm":
            # reflect achieved goal
            for t in range(len(traj["ag"])):
                obj_pos_in_central = traj["ag"][t][0:3] - self.central_point_in_global
                traj_invariant["ag"][t][0:3] = (self.bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                   .reshape(-1) + self.central_point_in_global
                original_r = R.from_quat(traj["ag"][t][3:7])
                traj_invariant["ag"][t][3:7] = (self.bias_r * original_r).as_quat()

        print('------start to check the difference------')
        min_error = 1000
        best_z = 0
        for k in range(1, 70):
            self.state_action_ratio_rotation = np.array([1, 5, k/10])
            self.action_state_ratio_rotation = np.array([1, 0.2, 10 / k])
            self.env.reset()
            self.env.env.sim.set_state(traj_invariant["sim_states"][0])
            self.env.goal = traj_invariant["goal"]
            self.env.env.goal = traj_invariant["goal"]
            timestep = 0
            total_error = 0
            for t in range(len(traj["actions"])):
                traj_invariant["actions"][t] = self.rotation_inv_action(traj["actions"][t])
            for action in traj_invariant["actions"]:
                self.env.env.sim.set_state(traj_invariant["sim_states"][timestep])
                obs = self.env.env._get_obs()
                self.env.step(action)
                timestep = timestep + 1
                real_state = self.env.env.sim.data.qpos
                total_error += np.linalg.norm(real_state - traj_invariant["sim_states"][timestep].qpos)
                # print(np.linalg.norm(real_state[30:33] - traj_invariant["sim_states"][timestep].qpos[30:33]))
            if total_error < min_error:
                min_error = total_error
                best_z = k / 10
            print('total_errors:', total_error, 'current_min_error: ', min_error,
                  'current_best_z: ', best_z)

    def test_action(self):
        print('---start testing---')
        test_len = 100
        state_dict = self.env.reset()
        for _ in range(test_len):
            action = np.zeros(self.env.action_space.shape[0])
            action[41] = -1
            state_dict, reward, done, _ = self.env.step(action)
            print(state_dict["observation"][60:63])
            # print(state_dict["observation"][0:3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchUnderarm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    transtest = CatchUnderarmTransformationTest(args)
    # transtest.test_translation()
    # transtest.test_rotation()
    transtest.test_action()