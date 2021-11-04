from invariant_traj_generator import InvariantTrajGenerator
import joblib
from copy import deepcopy
import numpy as np
import argparse
import random


# apply translation on the trajectory
class EggCatchUnderArmInvTrajGenerator(InvariantTrajGenerator):
    def generate_sym_traj(self):
        print('---start generating symmetric trajectories---')
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

            # reflect best rewards_it
            # reflect actions
            for t in range(len(traj["actions"])):
                # hand_1_mount
                traj_invariant["actions"][t][46:49] = traj["actions"][t][46:49] + hand_xyz_bias*np.array([5, 5, 12.5])
                # hand_2_mount
                traj_invariant["actions"][t][40:43] = traj["actions"][t][40:43] + hand_xyz_bias*np.array([-5, 5, -12.5])
            # reflect goal
            traj_invariant["goal"][0:3] = traj["goal"][0:3] + np.array([-1, -1, -1]) * obj_xyz_bias
            traj_list.append(traj)
            traj_invariant_list.append(traj_invariant)
            joblib.dump(traj_invariant, self.invariant_prefix + file)

        print('---stop generating symmetric trajectories---')
        return traj_list, traj_invariant_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="EggCatchUnderarm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    invtrajgen = EggCatchUnderArmInvTrajGenerator(args)
    traj_list, invariant_traj_list = invtrajgen.generate_sym_traj()
    # invtrajgen.inv_traj_render(traj_list[1], invariant_traj_list[1], False)

    invtrajgen.inv_traj_render(traj_list[0], invariant_traj_list[0], True)

    invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[1], True)
    # for i in range(0, 10):
    #     invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[i], True)
    # for i in range(0, 10):
    #     invtrajgen.compare_real_state_with_artificial_state(invariant_traj_list[i], False)
    # symtrajgen.recording(args, traj_list[0])
