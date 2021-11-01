from invariant_traj_generator import InvariantTrajGenerator
import joblib
from copy import deepcopy
import numpy as np
import argparse
import random
from scipy.spatial.transform import Rotation as R


# test applying transformation on the state and action
class TransformationTest(InvariantTrajGenerator):
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
        traj_name = self.traj_prefix + self.files[2]
        traj = joblib.load(traj_name)
        traj_invariant = deepcopy(traj)

        # reflect success (success is the same)
        # reflect rewards (rewards are the same)

        # apply rotation on the sim_states within a small range
        zrot_max_bias = 0.1
        # xyz_rot_bias = (np.random.rand(3)-0.5) * np.array([xrot_max_bias, yrot_max_bias, zrot_max_bias])
        xyz_rot_bias = np.array([0, 0, zrot_max_bias])
        bias_r = R.from_rotvec(xyz_rot_bias)
        bias_r_obj = R.from_rotvec(np.array([-zrot_max_bias, 0, 0]))
        bias_r_target = R.from_rotvec(np.array([0, -zrot_max_bias, 0]))

        central_point_in_global = np.array([1, 0.675, 0.15])
        mount_point_1_in_global = np.array([1, 1.35, 0.15])
        mount_point_2_in_global = np.array([1, 0, 0.15])
        bias_in_state_1 = np.zeros([len(traj["actions"]), 6])
        bias_in_state_2 = np.zeros([len(traj["actions"]), 6])
        for t in range(len(traj["sim_states"])):
            '''
            hand_1
            '''
            # hand_1_mount (position and rotation)
            # mount point position with respect to the central point
            mount_point_in_central = traj["sim_states"][t].qpos[0:3]+mount_point_1_in_global-central_point_in_global
            traj_invariant["sim_states"][t].qpos[0:3] = (bias_r.as_matrix()@mount_point_in_central.reshape([3, 1]))\
                                                    .reshape(-1)+central_point_in_global-mount_point_1_in_global

            traj_invariant["sim_states"][t].qpos[3:6] = bias_r.as_matrix()@\
                                        (traj["sim_states"][t].qpos[3:6]+xyz_rot_bias).reshape([3, 1]).reshape(-1)


            # save the bias for calculating the bias in action
            if t < len(traj["sim_states"])-1:
                bias_in_state_1[t, :3] = traj_invariant["sim_states"][t].qpos[0:3] - \
                                    bias_r.as_matrix()@(traj["sim_states"][t].qpos[0:3]).reshape([3, 1]).reshape(-1)
                bias_in_state_1[t, 3:6] = traj_invariant["sim_states"][t].qpos[3:6] - \
                                    bias_r.as_matrix()@(traj["sim_states"][t].qpos[3:6]).reshape([3, 1]).reshape(-1)
            '''
            hand_2
            '''
            # hand_2_mount (position and rotation)
            # mount point position with respect to the central point
            mount_point_in_central = np.array([-1, -1, 1])*traj["sim_states"][t].qpos[30:33]\
                                     +mount_point_2_in_global-central_point_in_global
            traj_invariant["sim_states"][t].qpos[30:33] = np.array([-1, -1, 1])*\
                ((bias_r.as_matrix()@mount_point_in_central.reshape([3, 1])).reshape(-1)
                 +central_point_in_global-mount_point_2_in_global)

            traj_invariant["sim_states"][t].qpos[33:36] = bias_r.as_matrix() @ \
                (traj["sim_states"][t].qpos[33:36] + xyz_rot_bias).reshape([3, 1]).reshape(-1)


            # save the bias for calculating the bias in action
            if t < len(traj["sim_states"])-1:
                bias_in_state_2[t, :3] = traj_invariant["sim_states"][t].qpos[30:33] - \
                                    bias_r.as_matrix()@(traj["sim_states"][t].qpos[30:33]).reshape([3, 1]).reshape(-1)
                bias_in_state_2[t, 3:6] = traj_invariant["sim_states"][t].qpos[33:36] - \
                                    bias_r.as_matrix()@(traj["sim_states"][t].qpos[33:36]).reshape([3, 1]).reshape(-1)
            '''
            obj
            '''
            # obj (position and quaternion)
            obj_pos_in_central = traj["sim_states"][t].qpos[60:63]-central_point_in_global
            traj_invariant["sim_states"][t].qpos[60:63] = (bias_r.as_matrix()@(obj_pos_in_central.reshape([3, 1])))\
                            .reshape(-1)+central_point_in_global

            original_r = R.from_quat(traj["sim_states"][t].qpos[63:67])
            traj_invariant["sim_states"][t].qpos[63:67] = (bias_r_obj*original_r).as_quat()

            # obj_target (position and quaternion)
            obj_pos_in_central = traj["sim_states"][t].qpos[67:70] - central_point_in_global
            traj_invariant["sim_states"][t].qpos[67:70] = (bias_r.as_matrix()@(obj_pos_in_central.reshape([3, 1])))\
                            .reshape(-1)+central_point_in_global

            original_r = R.from_quat(traj["sim_states"][t].qpos[70:74])
            traj_invariant["sim_states"][t].qpos[70:74] = (bias_r_target*original_r).as_quat()

        # reflect best rewards_it
        # reflect goal
        obj_pos_in_central = traj["goal"][0:3] - central_point_in_global
        traj_invariant["goal"][0:3] = (bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                          .reshape(-1) + central_point_in_global
        original_r = R.from_quat(traj["goal"][3:7])
        traj_invariant["goal"][3:7] = (bias_r_target * original_r).as_quat()
        # reflect achieved goal
        for t in range(len(traj["ag"])):
            obj_pos_in_central = traj["ag"][t][0:3] - central_point_in_global
            traj_invariant["ag"][t][0:3] = (bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                               .reshape(-1) + central_point_in_global
            original_r = R.from_quat(traj["ag"][t][3:7])
            traj_invariant["ag"][t][3:7] = (bias_r_obj * original_r).as_quat()

        print('------start to check the difference------')
        min_error = 1000
        best_x = 0
        best_y = 0
        best_z = 0
        for i in range(0, 1):
            for j in range(0, 1):
                for k in range(-60, 70):
                    self.env.reset()
                    self.env.env.sim.set_state(traj_invariant["sim_states"][0])
                    timestep = 0
                    total_error = 0
                    for t in range(len(traj["actions"])):
                        # hand_1_mount
                        traj_invariant["actions"][t][46:49] = bias_r.as_matrix() @ (traj["actions"][t][46:49]).reshape(
                            [3, 1]). \
                            reshape(-1) + bias_in_state_1[t, :3] * np.array([5, k/10, 12.5])
                        traj_invariant["actions"][t][49:52] = bias_r.as_matrix() @ (traj["actions"][t][49:52]).reshape(
                            [3, 1]). \
                            reshape(-1) + bias_in_state_1[t, 3:6] * np.array([1, 5, 5])
                        # hand_2_mount
                        traj_invariant["actions"][t][40:43] = bias_r.as_matrix() @ (traj["actions"][t][40:43]).reshape(
                            [3, 1]). \
                            reshape(-1) + bias_in_state_2[t, :3] * np.array([5, k/10, 12.5])
                        traj_invariant["actions"][t][43:46] = bias_r.as_matrix() @ (traj["actions"][t][43:46]).reshape(
                            [3, 1]). \
                            reshape(-1) + bias_in_state_2[t, 3:6] * np.array([1, 5, 5])
                    for action in traj_invariant["actions"]:
                        self.env.env.sim.set_state(traj_invariant["sim_states"][timestep])
                        self.env.step(action)
                        timestep = timestep + 1
                        real_state = self.env.env.sim.data.qpos
                        total_error += np.linalg.norm(real_state[0:6] - traj_invariant["sim_states"][timestep].qpos[0:6])
                        # print(np.linalg.norm(real_state[30:33] - traj_invariant["sim_states"][timestep].qpos[30:33]))
                    if total_error < min_error:
                        min_error = total_error
                        best_x = i / 1
                        best_y = j / 1
                        best_z = k / 10
                    print('total_errors:', total_error, 'current_min_error: ', min_error,
                          'current_best_x: ', best_x,
                          'current_best_y: ', best_y,
                          'current_best_z: ', best_z)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchUnderarm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    transtest = TransformationTest(args)
    transtest.test_translation()
    # transtest.test_rotation()