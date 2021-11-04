from invariant_traj_generator import InvariantTrajGenerator
import joblib
from copy import deepcopy
import numpy as np
import argparse
import random
from scipy.spatial.transform import Rotation as R


class EggCatchOverArmInvTrajGenerator(InvariantTrajGenerator):
    # apply translation on the trajectory
    def generate_translation_inv_traj(self):
        print('---start generating invariant trajectories---')
        traj_list = []
        traj_invariant_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_invariant = deepcopy(traj)

            # reflect success (success is the same)
            # reflect rewards (rewards are the same)

            # apply translation on the sim_states within a small range
            x_max_bias = 0.02
            y_max_bias = 0.02
            z_max_bias = 0.01
            # xyz_bias = (np.random.rand(3)-0.5) * np.array([x_max_bias, y_max_bias, z_max_bias])
            xyz_bias = np.array([x_max_bias, y_max_bias, z_max_bias])
            for t in range(len(traj["sim_states"])):
                '''
                hand_1
                '''
                # hand_1_mount (pos)
                traj_invariant["sim_states"][t].qpos[:3] = traj["sim_states"][t].qpos[:3] + xyz_bias

                '''
                hand_2
                '''
                # hand_2_mount (pos)
                traj_invariant["sim_states"][t].qpos[30:33] = traj["sim_states"][t].qpos[30:33] + \
                                                              np.array([-1, -1, 1]) * xyz_bias

                '''
                obj_1
                '''
                # obj_1 (pos)
                traj_invariant["sim_states"][t].qpos[60:63] = traj["sim_states"][t].qpos[60:63] + xyz_bias

                # obj_1_target (pos)
                traj_invariant["sim_states"][t].qpos[67:70] = traj["sim_states"][t].qpos[67:70] + xyz_bias

            # reflect best rewards_it
            # reflect actions
            for t in range(len(traj["actions"])):
                # hand_1_mount
                traj_invariant["actions"][t][46:49] = traj["actions"][t][46:49] + xyz_bias*np.array([5, 5, 12.5])
                # hand_2_mount
                traj_invariant["actions"][t][40:43] = traj["actions"][t][40:43] + xyz_bias*np.array([-5, -5, 12.5])
            # reflect goal
            traj_invariant["goal"][0:3] = traj["goal"][0:3] + xyz_bias
            # reflect achieved goal
            for t in range(len(traj["ag"])):
                traj_invariant["ag"][t][0:3] = traj["ag"][t][0:3] + xyz_bias
            traj_list.append(traj)
            traj_invariant_list.append(traj_invariant)
            joblib.dump(traj_invariant, self.invariant_prefix +file)

        print('---stop generating symmetric trajectories---')
        return traj_list, traj_invariant_list

    # apply rotation around the z-axis across the central point on the trajectory
    def generate_rotation_inv_traj(self):
        print('---start generating invariant trajectories---')
        traj_list = []
        traj_invariant_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_invariant = deepcopy(traj)

            # reflect success (success is the same)
            # reflect rewards (rewards are the same)

            # apply rotation on the sim_states within a small range
            zrot_max_bias = 0.05
            zrot_bias = (random.random()-0.5)*zrot_max_bias

            xyz_rot_bias = np.array([0, 0, zrot_bias])
            bias_r = R.from_rotvec(xyz_rot_bias)
            bias_r_obj = R.from_rotvec(np.array([-zrot_bias, 0, 0]))
            bias_r_target = R.from_rotvec(np.array([0, -zrot_bias, 0]))

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
            # reflect actions
            for t in range(len(traj["actions"])):
                # hand_1_mount
                traj_invariant["actions"][t][46:49] = bias_r.as_matrix()@(traj["actions"][t][46:49]).reshape([3, 1]).\
                    reshape(-1) + bias_in_state_1[t, :3]*np.array([5, 5, 12.5])
                traj_invariant["actions"][t][49:52] = bias_r.as_matrix()@(traj["actions"][t][49:52]).reshape([3, 1]). \
                    reshape(-1) + bias_in_state_1[t, 3:6]*np.array([1, 5, 5])
                # hand_2_mount
                traj_invariant["actions"][t][40:43] = (bias_r.as_matrix()@(traj["actions"][t][40:43]).reshape([3, 1]).\
                    reshape(-1)) + bias_in_state_2[t, :3]*np.array([5, 5, 12.5])
                traj_invariant["actions"][t][43:46] = bias_r.as_matrix()@(traj["actions"][t][43:46]).reshape([3, 1]). \
                    reshape(-1) + bias_in_state_2[t, 3:6]*np.array([1, 5, 5])

            # reflect goal
            obj_pos_in_central = traj["goal"][0:3] - central_point_in_global
            traj_invariant["goal"][0:3] = (bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                            .reshape(-1) + central_point_in_global

            original_r = R.from_quat(traj["goal"][3:7])
            traj_invariant["goal"][3:7] = (bias_r_target*original_r).as_quat()
            # reflect achieved goal
            for t in range(len(traj["ag"])):
                obj_pos_in_central = traj["ag"][t][0:3] - central_point_in_global
                traj_invariant["ag"][t][0:3] = (bias_r.as_matrix() @ (obj_pos_in_central.reshape([3, 1]))) \
                                                  .reshape(-1) + central_point_in_global
                original_r = R.from_quat(traj["ag"][t][3:7])
                traj_invariant["ag"][t][3:7] = (bias_r_obj*original_r).as_quat()
            traj_list.append(traj)
            traj_invariant_list.append(traj_invariant)
            joblib.dump(traj_invariant, self.invariant_prefix +file)

        print('---stop generating symmetric trajectories---')
        return traj_list, traj_invariant_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchOverarm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    invtrajgen = EggCatchOverArmInvTrajGenerator(args)
    traj_list, invariant_traj_list = invtrajgen.generate_rotation_inv_traj()
    # traj_list, invariant_traj_list = invtrajgen.generate_translation_inv_traj()
    # invtrajgen.inv_traj_render(traj_list[2], invariant_traj_list[2], False)
    invtrajgen.inv_traj_render(traj_list[3], invariant_traj_list[3], True)
    invtrajgen.inv_traj_render(traj_list[4], invariant_traj_list[4], True)
    invtrajgen.inv_traj_render(traj_list[5], invariant_traj_list[5], True)

    # trajgen.compare_real_state_with_artificial_state(traj_list[2])
    # invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[5], False)
    # symtrajgen.recording(args, traj_list[0])