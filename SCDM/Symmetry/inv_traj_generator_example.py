from invariant_traj_generator import InvariantTrajGenerator
from copy import deepcopy
import numpy as np
import argparse
import random
from scipy.spatial.transform import Rotation as R


class CatchUnderarmInvTrajGenerator(InvariantTrajGenerator):
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
        self.x_max_bias = 0.1
        self.y_max_bias = 0.1
        self.z_max_bias = 0.1

        # rotation max bias
        self.zrot_max_bias = 0.01

        # rotation info
        self.central_point_in_global = np.array([1, 0.675, 0.35])
        self.hand1_mount_point_in_global = np.array([1, 1.35, 0.35])
        self.hand2_mount_point_in_global = np.array([1, 0, 0.35])


class CatchOverarmInvTrajGenerator(InvariantTrajGenerator):
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
        self.tf_hand1_to_global = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
        self.tf_hand2_to_global = np.array([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]])

        self.tf_global_to_hand1 = np.linalg.inv(self.tf_hand1_to_global)
        self.tf_global_to_hand2 = np.linalg.inv(self.tf_hand2_to_global)

        # translation max bias
        self.x_max_bias = 0.1
        self.y_max_bias = 0.1
        self.z_max_bias = 0.1

        # rotation max bias
        self.zrot_max_bias = 0.01

        # rotation info
        self.central_point_in_global = np.array([1, 0.675, 0.15])
        self.hand1_mount_point_in_global = np.array([1, 1.35, 0.15])
        self.hand2_mount_point_in_global = np.array([1, 0, 0.15])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchOverarm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    invtrajgen = CatchOverarmInvTrajGenerator(args)
    #traj_list, invariant_traj_list = invtrajgen.generate_translation_inv_traj()
    model_file_1 = "../TD3_plus_demos/models/TD3_EggCatchOverarm-v0_0_random_goal_demo_2"
    model_file_2 = "../TD3_plus_demos/models/TD3_EggCatchOverarm-v0_1_random_goal_demo_exclude_demo_egg" \
                 "_in_the_air_add_hand_action_invariance_regularization_new"
    # invtrajgen.evaluate_transformed_policy(model_file)
    invtrajgen.evaluate_Q_value(model_file_1, model_file_2)
    # traj_list, invariant_traj_list = invtrajgen.generate_rotation_inv_traj()
    # invtrajgen.inv_traj_render(traj_list[1], invariant_traj_list[1], False)
    # invtrajgen.inv_traj_render(traj_list[2], invariant_traj_list[2], False)
    # invtrajgen.inv_traj_render(traj_list[3], invariant_traj_list[3], False)

    invtrajgen.inv_traj_render(traj_list[1], invariant_traj_list[1], True)

    # invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[1], True)
    for i in range(0, 10):
        invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[i], True)
    # for i in range(0, 10):
    #     invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[i], False)
    # symtrajgen.recording(args, traj_list[0])
