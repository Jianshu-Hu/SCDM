'''TwoObjectCatchUnderArm-v0
# State(qpos) Space #88
hand_1_mount_joint #6
hand_1_wrist_joint #2
hand_1_finger_joint #22

hand_2_mount_joint #6
hand_2_wrist_joint #2
hand_2_finger_joint #22

obj_1_joint #7
obj_1_target_joint #7
obj_2_joint #7
obj_2_target_joint #7


# Observation Space #146+#14
hand_1_mount_joint #6
hand_1_wrist_joint #2
hand_1_finger_joint #22

hand_1_mount_joint_vel #6
hand_1_wrist_joint_vel #2
hand_1_finger_joint_vel #22

hand_2_mount_joint #6
hand_2_wrist_joint #2
hand_2_finger_joint #22

hand_2_mount_joint_vel #6
hand_2_wrist_joint_vel #2
hand_2_finger_joint_vel #22

obj_1_joint #7
obj_1_joint_vel #6
obj_2_joint #7
obj_2_joint_vel #6

desired_obj_1_joint #7
desired_obj_2_joint #7

# Action Space #52
hand_1_wrist_joint #2
hand_1_finger_joint #18

hand_2_wrist_joint #2
hand_2_finger_joint #18

hand_2_mount_joint #6
hand_1_mount_joint #6
'''


'''OneObjectCatch-v0
# State(qpos) Space #74
hand_1_mount_joint #6
hand_1_wrist_joint #2
hand_1_finger_joint #22

hand_2_mount_joint #6
hand_2_wrist_joint #2
hand_2_finger_joint #22

# for overarm, obj in hand1
# for underarm, obj in hand2
obj_joint #7
obj_target_joint #7

# Observation Space #133+#7
hand_1_mount_joint #6
hand_1_wrist_joint #2
hand_1_finger_joint #22

hand_1_mount_joint_vel #6
hand_1_wrist_joint_vel #2
hand_1_finger_joint_vel #22

hand_2_mount_joint #6
hand_2_wrist_joint #2
hand_2_finger_joint #22

hand_2_mount_joint_vel #6
hand_2_wrist_joint_vel #2
hand_2_finger_joint_vel #22

obj_joint #7
obj_joint_vel #6

desired_obj_joint #7

# Action Space #52
hand_1_wrist_joint #2
hand_1_finger_joint #18

hand_2_wrist_joint #2
hand_2_finger_joint #18

hand_2_mount_joint #6
hand_1_mount_joint #6
'''
