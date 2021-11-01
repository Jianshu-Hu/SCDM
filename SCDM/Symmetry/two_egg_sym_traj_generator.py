from invariant_traj_generator import SymmetricTrajGenerator
import joblib
from copy import deepcopy
import numpy as np
import argparse


class TwoEggCatchUnderArmSymTrajGenerator(SymmetricTrajGenerator):
    def generate_sym_traj(self):
        print('---start generating symmetric trajectories---')
        traj_list = []
        traj_reflected_list = []
        for file in self.files:
            traj_name = self.traj_prefix + file
            traj = joblib.load(traj_name)
            traj_reflected = deepcopy(traj)

            # reflect success (success is the same)
            # reflect rewards (rewards are the same)
            # rorate sim_states by (x=1, y=0.675)
            for t in range(len(traj["sim_states"])):
                '''
                hand_1
                '''
                # hand_1_mount (pos & vel)
                traj_reflected["sim_states"][t].qpos[:8] = traj["sim_states"][t].qpos[30:38]
                traj_reflected["sim_states"][t].qvel[:8] = traj["sim_states"][t].qvel[30:38]
                # hand_1_finger (joints & vel)
                traj_reflected["sim_states"][t].qpos[8:30] = traj["sim_states"][t].qpos[38:60]
                traj_reflected["sim_states"][t].qvel[8:30] = traj["sim_states"][t].qvel[38:60]

                '''
                hand_2
                '''
                # hand_2_mount (pos & vel)
                traj_reflected["sim_states"][t].qpos[30:38] = traj["sim_states"][t].qpos[:8]
                traj_reflected["sim_states"][t].qvel[30:38] = traj["sim_states"][t].qvel[:8]
                # hand_2_finger (joints & vel)
                traj_reflected["sim_states"][t].qpos[38:60] = traj["sim_states"][t].qpos[8:30]
                traj_reflected["sim_states"][t].qvel[38:60] = traj["sim_states"][t].qvel[8:30]

                '''
                obj_1
                '''
                # obj_1 (pos & vel)
                traj_reflected["sim_states"][t].qpos[60:63] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                              traj["sim_states"][t].qpos[74:77]
                traj_reflected["sim_states"][t].qvel[60:63] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[72:75]
                # obj_1 (rot & ang_vel)
                traj_reflected["sim_states"][t].qpos[63:67] = np.array([1, -1, -1, 1]) * traj["sim_states"][t].qpos[
                                                                                         77:81]
                traj_reflected["sim_states"][t].qvel[63:66] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[75:78]

                # obj_1_target (pos & vel)
                traj_reflected["sim_states"][t].qpos[67:70] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                              traj["sim_states"][t].qpos[81:84]
                traj_reflected["sim_states"][t].qvel[66:69] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[78:81]
                # obj_1_target (rot & ang_vel)
                traj_reflected["sim_states"][t].qpos[70:74] = np.array([1, -1, -1, 1]) * traj["sim_states"][t].qpos[
                                                                                         84:88]
                traj_reflected["sim_states"][t].qvel[69:72] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[81:84]

                '''
                obj_2
                '''
                # obj_2 (pos & vel)
                traj_reflected["sim_states"][t].qpos[74:77] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                              traj["sim_states"][t].qpos[60:63]
                traj_reflected["sim_states"][t].qvel[72:75] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[60:63]
                # obj_2 (rot & ang_vel)
                traj_reflected["sim_states"][t].qpos[77:81] = np.array([1, -1, -1, 1]) * traj["sim_states"][t].qpos[
                                                                                         63:67]
                traj_reflected["sim_states"][t].qvel[75:78] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[63:66]

                # obj_2_target (pos & vel)
                traj_reflected["sim_states"][t].qpos[81:84] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                              traj["sim_states"][t].qpos[67:70]
                traj_reflected["sim_states"][t].qvel[78:81] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[66:69]
                # obj_2_target (rot & ang_vel)
                traj_reflected["sim_states"][t].qpos[84:88] = np.array([1, -1, -1, 1]) * traj["sim_states"][t].qpos[
                                                                                         70:74]
                traj_reflected["sim_states"][t].qvel[81:84] = np.array([-1, -1, 1]) * traj["sim_states"][t].qvel[69:72]

            # reflect best rewards_it
            # reflect actions
            for t in range(len(traj["actions"])):
                # exchange joints of fingers and wrist joints
                traj_reflected["actions"][t][:20] = traj["actions"][t][20:40]
                traj_reflected["actions"][t][20:40] = traj["actions"][t][:20]
                # exchange pos and rot of mount
                traj_reflected["actions"][t][40:46] = traj["actions"][t][46:52]
                traj_reflected["actions"][t][46:52] = traj["actions"][t][40:46]

            # reflect goal
            traj_reflected["goal"]['object_1'][0:3] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                      traj["goal"]['object_2'][0:3]
            traj_reflected["goal"]['object_1'][3:7] = np.array([1, -1, -1, 1]) * traj["goal"]['object_2'][3:7]

            traj_reflected["goal"]['object_2'][0:3] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                      traj["goal"]['object_1'][0:3]
            traj_reflected["goal"]['object_2'][3:7] = np.array([1, -1, -1, 1]) * traj["goal"]['object_1'][3:7]

            # reflect achieved goal
            for t in range(len(traj["ag"])):
                traj_reflected["ag"][t]['object_1'][0:3] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                          traj["ag"][t]['object_2'][0:3]
                traj_reflected["ag"][t]['object_1'][3:7] = np.array([1, -1, -1, 1]) * traj["ag"][t]['object_2'][3:7]

                traj_reflected["ag"][t]['object_2'][0:3] = np.array([2, 1.35, 0]) + np.array([-1, -1, 1]) * \
                                                          traj["ag"][t]['object_1'][0:3]
                traj_reflected["ag"][t]['object_2'][3:7] = np.array([1, -1, -1, 1]) * traj["ag"][t]['object_1'][3:7]

            traj_list.append(traj)
            traj_reflected_list.append(traj_reflected)
            joblib.dump(traj_reflected, self.symmetric_prefix +file)
        print('---stop generating symmetric trajectories---')
        return traj_list, traj_reflected_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="TwoEggCatchUnderArm-v0")
    parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
    args = parser.parse_args()

    symtrajgen = TwoEggCatchUnderArmSymTrajGenerator(args)
    traj_list, reflected_traj_list = symtrajgen.generate_sym_traj()
    symtrajgen.sym_traj_render(traj_list[2], reflected_traj_list[2], False)
    # symtrajgen.sym_traj_render(traj_list[2], reflected_traj_list[2], True)

    # symtrajgen.compare_real_state_with_artificial_state(traj_list[0])
    # symtrajgen.compare_real_state_with_artificial_state(reflected_traj_list[2])
    # symtrajgen.recording(args, traj_list[0])