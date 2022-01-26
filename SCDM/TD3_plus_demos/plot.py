import numpy as np
import matplotlib.pyplot as plt
import math
overarm_prefix = "TD3_EggCatchOverarm-v0_"
overarm_prefix_v1 = "TD3_EggCatchOverarm-v1_"
overarm_prefix_v2 = "TD3_EggCatchOverarm-v2_"
overarm_prefix_v3 = "TD3_EggCatchOverarm-v3_"

underarm_prefix = "TD3_EggCatchUnderarm-v0_"
underarmhard_prefix = "TD3_EggCatchUnderarmHard-v0_"
underarm_prefix_v1 = "TD3_EggCatchUnderarm-v1_"
underarm_prefix_v3 = "TD3_EggCatchUnderarm-v3_"
# overarm and underarm
tag_1 = ["0_random_goal_demo_1", "0_random_goal_demo_2", "0_random_goal_demo_3", "0_random_goal_demo_4", "0_random_goal_demo_5"]
tag_2 = ["random_goal_demo_with_normalizer_1", "random_goal_demo_with_normalizer_2", "random_goal_demo_with_normalizer_3"]

tag_7 = ["1_random_goal_demo_pr_init_0", "2_random_goal_demo_pr_init_0", "3_random_goal_demo_pr_init_0"]
tag_8 = ["1_random_goal_demo_pr_init_03", "2_random_goal_demo_pr_init_03", "3_random_goal_demo_pr_init_03"]

tag_3 = ["1_random_goal_demo_her_type_1_goal_from_demo", "2_random_goal_demo_her_type_1_goal_from_demo",
          "3_random_goal_demo_her_type_1_goal_from_demo"]
tag_4 = ["1_random_goal_demo_her_type_2_goal_from_demo", "2_random_goal_demo_her_type_2_goal_from_demo",
          "3_random_goal_demo_her_type_2_goal_from_demo"]

tag_5 = ["0_1_random_goal_demo_with_translation_regularization", "0_2_random_goal_demo_with_translation_regularization",
          "0_3_random_goal_demo_with_translation_regularization"]
tag_6 = ["0_1_random_goal_demo_with_rotation_regularization", "0_2_random_goal_demo_with_rotation_regularization",
          "0_3_random_goal_demo_with_rotation_regularization"]

tag_9 = ["1_random_goal_demo_her_type_1_goal_from_demo_successful_demos",
         "2_random_goal_demo_her_type_1_goal_from_demo_successful_demos",
         "3_random_goal_demo_her_type_1_goal_from_demo_successful_demos"]

tag_10 = ["0_1_random_goal_demo_replaced_by_001_translation", "0_2_random_goal_demo_replaced_by_001_translation",
         "0_3_random_goal_demo_replaced_by_001_translation"]
tag_11 = ["1_random_goal_demo_replaced_by_001_rotation", "2_random_goal_demo_replaced_by_001_rotation",
         "3_random_goal_demo_replaced_by_001_rotation"]

tag_12 = ["0_1_random_goal_demo_replaced_by_informative_001_translation", "0_2_random_goal_demo_replaced_by_informative_001_translation",
         "0_3_random_goal_demo_replaced_by_informative_001_translation"]
tag_13 = ["1_random_goal_demo_replaced_by_informative_001_rotation", "2_random_goal_demo_replaced_by_informative_001_rotation",
         "3_random_goal_demo_replaced_by_informative_001_rotation"]

tag_14 = ["0_1_random_goal_demo_with_001_translation", "0_2_random_goal_demo_with_001_translation",
         "0_3_random_goal_demo_with_001_translation"]
tag_15 = ["1_random_goal_demo_with_001_rotation", "2_random_goal_demo_with_001_rotation",
         "3_random_goal_demo_with_001_rotation"]

tag_16 = ["0_1_random_goal_demo_with_translation_regularization_2_artifical_samples",
          "0_2_random_goal_demo_with_translation_regularization_2_artifical_samples",
          "0_3_random_goal_demo_with_translation_regularization_2_artifical_samples"]
tag_17 = ["0_1_random_goal_demo_with_rotation_regularization_2_artifical_samples",
          "0_2_random_goal_demo_with_rotation_regularization_2_artifical_samples",
          "0_3_random_goal_demo_with_rotation_regularization_2_artifical_samples"]

tag_18 = ["0_1_random_goal_demo_with_01_translation", "0_2_random_goal_demo_with_01_translation",
          "0_3_random_goal_demo_with_01_translation"]

tag_19 = ["0_1_random_goal_demo_larger_workspace", "0_2_random_goal_demo_larger_workspace", "0_3_random_goal_demo_larger_workspace"]
tag_20 = ["0_1_random_goal_demo_larger_workspace_with_01_translation", "0_2_random_goal_demo_larger_workspace_with_01_translation",
          "0_3_random_goal_demo_larger_workspace_with_01_translation"]

tag_21 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air", "0_2_random_goal_demo_exclude_demo_egg_in_the_air",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air"]

tag_22 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_1e5_buffer_size"]
tag_23 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size"]
tag_24 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss",
          "0_2_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss"]

tag_25 = ["0_1_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss",
          "0_2_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss",
          "0_3_random_goal_demo_exclude_demo_egg_in_the_air_5e5_buffer_size_add_bc_loss"]

# tag_26 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air",
#           "0_2_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air",
#           "0_3_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air"]
tag_26 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air",
          "0_3_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air"]

# tag_27 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization",
#           "2_2_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization",
#           "3_3_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization"]
tag_27 = ["0_1_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization",
          "2_2_random_goal_demo_larger_workspace_exclude_demo_egg_in_the_air_translation_regularization"]

tag_28 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_initialize_with_demo"]

tag_29 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state"]

tag_30 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation"]

tag_31 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_larger_critic_lr",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_larger_critic_lr",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_larger_critic_lr"]

tag_32 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_4e-4_critic_lr",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_4e-4_critic_lr",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_bc_loss_4e-4_critic_lr"]

tag_33 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_translated_demos",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_translated_demos",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_translated_demos"]

tag_34 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos"]


tag_35 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation"]
tag_36 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation_regularization",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation_regularization",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_with_translation_regularization"]

tag_37 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_regularization",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_regularization",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_regularization"]

tag_38 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_samples",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_samples",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_translation_samples"]

tag_39 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss"]

tag_40 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss_without_filter",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss_without_filter",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_random_initial_state_new_demos_with_bc_loss_without_filter"]

tag_41 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization"]

tag_42 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples"]

tag_43 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer"]

tag_44 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer_add_bc_loss_without_filter",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer_add_bc_loss_without_filter",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_with_normalizer_add_bc_loss_without_filter"]

tag_45 = ["0_random_goal_demo_with_normalizer_1",
          "0_random_goal_demo_with_normalizer_2",
          "0_random_goal_demo_with_normalizer_3"]

tag_46 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_use_invariance_in_policy",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_use_invariance_in_policy",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_use_invariance_in_policy"]

tag_47 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples_more_actions",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples_more_actions",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_more_samples_more_actions"]

tag_48 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_slower",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_slower",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_slower"]

tag_49 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_faster",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_faster",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_throw_decay_faster"]
tag_50 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_fix_throw_prob",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_fix_throw_prob",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_fix_throw_prob"]

tag_51 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new"]

tag_52 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_average_target",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_average_target",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_average_target"]

tag_53 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_add_policy_penalty",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_add_policy_penalty",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_hand_action_invariance_regularization_new_add_policy_penalty"]

tag_54 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_add_two_hand_action_invariance_regularization_new",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_add_two_hand_action_invariance_regularization_new",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_add_two_hand_action_invariance_regularization_new"]

tag_55 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_fix_larger_throw_prob",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_fix_larger_throw_prob",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_fix_larger_throw_prob"]

tag_56 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob"]

tag_57 = ["1_random_goal_demo_demo_divided_into_two_part",
          "2_random_goal_demo_demo_divided_into_two_part",
          "3_random_goal_demo_demo_divided_into_two_part"]

tag_58 = ["1_random_goal_demo_only_exclude_demo_egg_in_the_air",
          "2_random_goal_demo_only_exclude_demo_egg_in_the_air",
          "3_random_goal_demo_only_exclude_demo_egg_in_the_air"]

tag_59 = ["1_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob_with_normalizer",
          "2_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob_with_normalizer",
          "3_random_goal_demo_exclude_demo_egg_in_the_air_fix_smaller_throw_prob_with_normalizer"]

tag_60 = ["1_random_goal_demo_demo_divided_into_two_part_catch_first",
          "2_random_goal_demo_demo_divided_into_two_part_catch_first",
          "3_random_goal_demo_demo_divided_into_two_part_catch_first"]

tag_61 = ["1_random_goal_demo_demo_divided_into_two_part_add_bc_loss_with_Q_filter",
          "2_random_goal_demo_demo_divided_into_two_part_add_bc_loss_with_Q_filter",
          "3_random_goal_demo_demo_divided_into_two_part_add_bc_loss_with_Q_filter"]
tag_62 = ["1_random_goal_demo_demo_divided_into_two_part_catch_first_add_bc_loss_with_Q_filter",
          "2_random_goal_demo_demo_divided_into_two_part_catch_first_add_bc_loss_with_Q_filter",
          "3_random_goal_demo_demo_divided_into_two_part_catch_first_add_bc_loss_with_Q_filter"]

tag_63 = ["1_random_goal_demo_demo_divided_into_two_part_average_target",
          "2_random_goal_demo_demo_divided_into_two_part_average_target",
          "3_random_goal_demo_demo_divided_into_two_part_average_target"]

tag_64 = ["1_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization_larger_threshold",
          "2_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization_larger_threshold",
          "3_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization_larger_threshold"]

tag_65 = ["1_random_goal_demo_demo_divided_into_two_part_add_policy_penalty_all_actions",
          "2_random_goal_demo_demo_divided_into_two_part_add_policy_penalty_all_actions",
          "3_random_goal_demo_demo_divided_into_two_part_add_policy_penalty_all_actions"]
tag_66 = ["1_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization",
          "2_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization",
          "3_random_goal_demo_demo_divided_into_two_part_add_two_hand_action_invariance_regularization"]
tag_67 = ["1_random_goal_demo_demo_divided_into_two_part_only_long_demo",
          "2_random_goal_demo_demo_divided_into_two_part_only_long_demo",
          "3_random_goal_demo_demo_divided_into_two_part_only_long_demo"]
tag_68 = ["1_random_goal_demo_demo_divided_into_two_part_only_short_demo",
          "2_random_goal_demo_demo_divided_into_two_part_only_short_demo",
          "3_random_goal_demo_demo_divided_into_two_part_only_short_demo"]

tag_69 = ["4_random_goal_demo_demo_divided_into_two_part_debug_Q_at_50",
          "5_random_goal_demo_demo_divided_into_two_part_debug_Q_at_50",
          "6_random_goal_demo_demo_divided_into_two_part_debug_Q_at_50"]

tag_70 = ["1_random_goal_demo_demo_divided_into_two_part_add_hand_action_invariance_regularization_larger_threshold",
          "2_random_goal_demo_demo_divided_into_two_part_add_hand_action_invariance_regularization_larger_threshold",
          "3_random_goal_demo_demo_divided_into_two_part_add_hand_action_invariance_regularization_larger_threshold"]
tag_71 = ["1_random_goal_demo","2_random_goal_demo","3_random_goal_demo"]

tag_72 = ["1_random_goal_demo_demo_divided_into_two_part_average_target_larger_threshold",
          "2_random_goal_demo_demo_divided_into_two_part_average_target_larger_threshold",
          "3_random_goal_demo_demo_divided_into_two_part_average_target_larger_threshold"]

tag_73 = ["1_random_goal_demo_demo_divided_into_two_part_add_auto_regularization",
          "2_random_goal_demo_demo_divided_into_two_part_add_auto_regularization",
          "3_random_goal_demo_demo_divided_into_two_part_add_auto_regularization"]
tag_74 = ["1_random_goal_demo_demo_divided_into_two_part_add_auto_regularization_add_policy_penalty_all_actions",
          "2_random_goal_demo_demo_divided_into_two_part_add_auto_regularization_add_policy_penalty_all_actions",
          "3_random_goal_demo_demo_divided_into_two_part_add_auto_regularization_add_policy_penalty_all_actions"]

tag_75 = ["1_random_goal_demo_learn_transition", "2_random_goal_demo_learn_transition", "3_random_goal_demo_learn_transition"]

tag_76 = ["1_random_goal_demo_add_artificial_transitions",
          "2_random_goal_demo_add_artificial_transitions",
          "3_random_goal_demo_add_artificial_transitions"]

# underarm
# tag_9 = ["1_random_goal_demo_her_type_4_segment", "2_random_goal_demo_her_type_4_segment",
#          "3_random_goal_demo_her_type_4_segment"]
# tag_10 = ["1_random_goal_demo_her_type_5_segment", "2_random_goal_demo_her_type_5_segment",
#          "3_random_goal_demo_her_type_5_segment"]
# tag_4 = ["1_random_goal_demo_with_0001_translation", "2_random_goal_demo_with_0001_translation",
#          "3_random_goal_demo_with_0001_translation"]
# tag_5 = ["1_random_goal_demo_with_0001_rotation", "2_random_goal_demo_with_0001_rotation",
#          "3_random_goal_demo_with_0001_rotation"]
# tag_25 = ["1_random_goal_demo_replaced_by_informative_001_translation", "2_random_goal_demo_replaced_by_informative_001_translation"]
# tag_12 = ["1_random_goal_demo_her_type_4_full_trajectory_pd_init_08", "2_random_goal_demo_her_type_4_full_trajectory_pd_init_08",
#          "3_random_goal_demo_her_type_4_full_trajectory_pd_init_08"]
# tag_13 = ["1_random_goal_demo_her_type_5_full_trajectory_pd_init_08", "2_random_goal_demo_her_type_5_full_trajectory_pd_init_08",
#          "3_random_goal_demo_her_type_5_full_trajectory_pd_init_08"]
# tag_17 = ["1_random_goal_demo_her_type_4_full_trajectory_start_1e6", "2_random_goal_demo_her_type_4_full_trajectory_start_1e6",
#           "3_random_goal_demo_her_type_4_full_trajectory_start_1e6"]
# tag_18 = ["1_random_goal_demo_her_type_4_full_trajectory_start_2e6", "2_random_goal_demo_her_type_4_full_trajectory_start_2e6",
#           "3_random_goal_demo_her_type_4_full_trajectory_start_2e6"]
# tag_21 = ["1_random_goal_demo_her_type_4_full_trajectory_sparse_reward", "2_random_goal_demo_her_type_4_full_trajectory_sparse_reward",
#           "3_random_goal_demo_her_type_4_full_trajectory_sparse_reward"]
# tag_22 = ["1_random_goal_demo_sparse_reward", "2_random_goal_demo_sparse_reward", "3_random_goal_demo_sparse_reward"]
# tag_27 = ["1_random_goal_demo_her_type_1_goal_from_demo", "2_random_goal_demo_her_type_1_goal_from_demo",
#           "3_random_goal_demo_her_type_1_goal_from_demo"]
# tag_28 = ["1_random_goal_demo_her_type_2_goal_from_demo", "2_random_goal_demo_her_type_2_goal_from_demo",
#           "3_random_goal_demo_her_type_2_goal_from_demo"]
# # overarm
# tag_7 = ["1_random_goal_demo_with_informative_001_translation", "2_random_goal_demo_with_informative_001_translation",
#          "3_random_goal_demo_with_informative_001_translation"]
# tag_11 = ["1_random_goal_demo_with_informative_001_rotation", "2_random_goal_demo_with_informative_001_rotation",
#          "3_random_goal_demo_with_informative_001_rotation"]
# tag_15 = ["1_random_goal_demo_replaced_by_informative_001_translation", "2_random_goal_demo_replaced_by_informative_001_translation",
#          "3_random_goal_demo_replaced_by_informative_001_translation"]
# tag_19 = ["1_random_goal_demo_replaced_by_informative_001_translation_include_original_sample",
#           "2_random_goal_demo_replaced_by_informative_001_translation_include_original_sample",
#           "3_random_goal_demo_replaced_by_informative_001_translation_include_original_sample"]
# tag_20 = ["1_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample",
#           "2_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample",
#           "3_random_goal_demo_replaced_by_informative_001_rotation_include_original_sample"]
# tag_33 = ["1_random_goal_demo_with_translation_regularization_new", "2_random_goal_demo_with_translation_regularization_new",
#           "3_random_goal_demo_with_translation_regularization_new"]
# tag_34 = ["1_random_goal_demo_with_rotation_regularization_new", "2_random_goal_demo_with_rotation_regularization_new",
#           "3_random_goal_demo_with_rotation_regularization_new"]


def plot_all_fig(prefix=underarm_prefix, tag=tag_2, plot_or_save='save'):
    fig, axs = plt.subplots(2, 1)
    data_list = []
    min_len = np.inf
    for i in range(len(tag)):
        data = np.load('results/'+prefix + tag[i] + ".npy")
        data_list.append(data)
        min_len = min(min_len, data.shape[0])

        axs[0].plot(range(len(data)), data, label=tag[i])
        axs[0].set_xlabel('timesteps/5000')
        axs[0].set_ylabel('average rewards')
        axs[0].legend()

    sum = np.zeros(min_len)
    for i in range(len(tag)):
        sum = sum + data_list[i][:min_len]
    average = sum/len(tag)
    axs[1].plot(range(len(average)), average, label='Average')
    axs[1].set_xlabel('timesteps/5000')
    axs[1].set_ylabel('average rewards')
    axs[1].legend()
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_all')


def average_over_experiments(prefix, tag):
    data_list = []
    min_len = np.inf
    for i in range(len(tag)):
        data = np.load('results/'+prefix + tag[i] + ".npy")
        data_list.append(data)
        min_len = min(min_len, data.shape[0])
    data_np = np.zeros([len(tag), min_len])
    for i in range(len(tag)):
        data_np[i, :] = np.copy(data_list[i][:min_len])
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0)
    return mean, std


def compare(prefix, tag_list, title='', label_list=[""], plot_or_save='save'):
    plt.rcParams['figure.figsize'] = (10, 6)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(tag_list)):
        mean, std = average_over_experiments(prefix, tag_list[i])
        # plot variance
        axs.fill_between(range(len(mean)), mean - std/math.sqrt(len(tag_list)), mean + std/math.sqrt(len(tag_list)),
                         alpha=0.4)
        # axs.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        # plot mean
        if len(label_list) == len(tag_list):
            # specify label
            axs.plot(range(len(mean)), mean, label=label_list[i])
        else:
            axs.plot(range(len(mean)), mean, label=tag_list[i][0])
    axs.set_title(prefix+title)
    axs.set_xlabel('timesteps/5000')
    axs.set_ylabel('average rewards')
    axs.legend()
    if plot_or_save=='plot':
        plt.show()
    else:
        plt.savefig('saved_fig/'+prefix+title)


def compare_policy_critic(prefix, prefix_critic, tag, plot_or_save='save'):
    fig, axs = plt.subplots(len(tag), 1)
    for i in range(len(tag)):
        data1 = np.load('results/'+prefix + tag[i] + ".npy")
        data2 = np.load('results_critic/'+prefix_critic + tag[i] + ".npy")
        axs[i].plot(range(len(data1)), data1, label="policy")
        axs[i].plot(range(len(data2)), data2, label="critic")
        axs[i].set_xlabel('timesteps/5000')
        axs[i].set_ylabel('average rewards')
        axs[i].legend()
    axs[0].set_title(prefix+tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_policy_critic')


def plot_actor_critic_loss(prefix=overarm_prefix, tag=tag_59, plot_or_save='save'):
    fig, axs = plt.subplots(2*len(tag), 1)
    for i in range(len(tag)):
        critic_loss = np.load('results/'+prefix + tag[i] + "_critic_loss.npy")
        actor_loss = np.load('results/'+prefix + tag[i] + "_actor_loss.npy")
        axs[2*i].plot(range(len(critic_loss)), critic_loss, label="critic_loss")
        axs[2*i+1].plot(range(len(actor_loss)), actor_loss, label="actor_loss")

        axs[2*i].set_xlabel('timesteps')
        axs[2*i].set_ylabel('critic loss')
        axs[2*i].legend()

        axs[2*i+1].set_xlabel('timesteps/2')
        axs[2*i+1].set_ylabel('actor loss')
        axs[2 * i+1].legend()

        axs[0].set_title(prefix + tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_actor_critic_loss')


def plot_transition_model_loss(prefix=overarm_prefix, tag=tag_59, plot_or_save='save'):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(tag)):
        loss = np.load('results/'+prefix + tag[i] + "_transition_model_loss.npy")
        axs.plot(range(len(loss)), loss, label=tag[i])

        axs.set_xlabel('timesteps')
        axs.set_ylabel('model loss')
        axs.legend()
        axs.set_title(prefix + tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_transition_loss')

#labellist = ["random_initial_state", "random_initial_state_new_demos", "with_translation", "with_translation_regularization"]
#compare(prefix=overarm_prefix_v3, tag_list=[tag_29, tag_34, tag_35, tag_37], title="new_demos", label_list=labellist)

# compare_policy_critic(underarm_prefix_v3, underarm_critic_prefix_v3, tag_33)
# compare_policy_critic(overarm_prefix_v3, overarm_critic_prefix_v3, tag_30)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_44)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_48)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_50)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_55)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_56)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_57)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_58)
# compare_policy_critic(overarm_prefix, overarm_critic_prefix, tag_32)
# compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_264
# compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_20)
# compare_policy_critic(underarm_prefix_v1, underarm_critic_prefix_v1, tag_27)
# baseline
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_21, tag_22, tag_23, tag_24], title="baseline")
# plot_loss(prefix=overarm_prefix, tag=tag_59)
# plot_loss(prefix=overarm_prefix, tag=tag_60)
#plot_all_fig(prefix=overarm_prefix, tag=tag_57)
#plot_all_fig(prefix=overarm_prefix, tag=tag_59)
#plot_all_fig(prefix=overarm_prefix, tag=tag_60)
#plot_all_fig(prefix=overarm_prefix, tag=tag_61)
#plot_all_fig(prefix=overarm_prefix, tag=tag_62)
#plot_all_fig(prefix=underarm_prefix, tag=tag_67)
plot_all_fig(prefix=underarm_prefix, tag=tag_75)
plot_transition_model_loss(prefix=underarm_prefix, tag=tag_75)
plot_all_fig(prefix=overarm_prefix, tag=tag_75)
plot_transition_model_loss(prefix=overarm_prefix, tag=tag_75)
plot_all_fig(prefix=underarmhard_prefix, tag=tag_71)
compare(prefix=underarmhard_prefix, tag_list=[tag_71], title="baseline")

compare(prefix=overarm_prefix, tag_list=[tag_1, tag_76], title="add_transitions")
compare(prefix=underarm_prefix, tag_list=[tag_1, tag_76], title="add_transitions")

compare(prefix=underarm_prefix, tag_list=[tag_1, tag_57, tag_63, tag_64, tag_65], title="tuned_hand_invariance")
compare(prefix=underarm_prefix, tag_list=[tag_57, tag_73, tag_74], title="auto_hand_invariance")

compare(prefix=overarm_prefix, tag_list=[tag_1, tag_57, tag_70, tag_65, tag_66, tag_72], title="tuned_hand_invariance")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_57, tag_73, tag_74], title="auto_hand_invariance")
compare(prefix=underarm_prefix, tag_list=[tag_57, tag_67, tag_68], title="demo_len")
plot_all_fig(prefix=overarm_prefix, tag=tag_69)
compare_policy_critic(overarm_prefix, overarm_prefix, tag_69)

compare(prefix=underarm_prefix, tag_list=[tag_1, tag_21, tag_43, tag_24, tag_28], title="baseline")
compare(prefix=underarm_prefix, tag_list=[tag_21, tag_51, tag_52, tag_53, tag_54], title="new_hand_invariance")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_22, tag_23, tag_24, tag_25], title="baseline")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_45, tag_21, tag_24, tag_43, tag_44], title="baseline")
compare(prefix=overarm_prefix, tag_list=[tag_21, tag_48, tag_49, tag_50, tag_55, tag_56, tag_59], title="decay rate")
compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_57, tag_58, tag_60, tag_61, tag_62], title="exclude demo")
# compare(prefix=overarm_prefix, tag_list=[tag_21, tag_41, tag_42, tag_46, tag_47], title="hand_invariance")
compare(prefix=overarm_prefix, tag_list=[tag_21, tag_51, tag_52, tag_53,tag_54], title="new_hand_invariance")
# compare(prefix=overarm_prefix, tag_list=[tag_24, tag_32, tag_31], title="critic_lr")
# plot_all_fig(prefix=overarm_prefix, tag=tag_1)
# plot_all_fig(prefix=overarm_prefix, tag=tag_21)
# plot_all_fig(prefix=overarm_prefix, tag=tag_24)

# compare(prefix=underarm_prefix_v1, tag_list=[tag_19, tag_26, tag_20, tag_27], title="baseline")
# compare(prefix=overarm_prefix_v1, tag_list=[tag_29, tag_30, tag_33], title="baseline")
# compare(prefix=overarm_prefix_v2, tag_list=[tag_29, tag_30, tag_33], title="baseline")
# compare(prefix=overarm_prefix_v3, tag_list=[tag_29, tag_30, tag_36, tag_33], title="baseline")
# compare(prefix=overarm_prefix_v3, tag_list=[tag_29, tag_34, tag_35, tag_37, tag_38], title="new_demos")
# compare(prefix=overarm_prefix_v3, tag_list=[tag_34, tag_39, tag_40], title="new_demos_add_bc_loss")
# compare(prefix=underarm_prefix_v3, tag_list=[tag_29, tag_30, tag_36, tag_33], title="baseline")


# # translation/rotation
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_10, tag_12, tag_14, tag_18], title="translation")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_11, tag_13, tag_15], title="rotation")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_10, tag_12, tag_14, tag_18], title="translation")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_11, tag_13, tag_15], title="rotation")

# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_5, tag_6], title="regularization")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_5, tag_6], title="regularization")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_16, tag_17], title="regularization")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_16, tag_17], title="regularization",
#         label_list=["learning from demo", "001 translation regularization averaged by 2 samples",
#                     "001 rotation regularization averaged by 2 samples"])

# # HER
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_3, tag_4], title='HER_goal_from_demo')
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_3, tag_4, tag_9], title='HER_goal_from_demo')

# plot with specific label
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_2, tag_3], title="HER")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_2, tag_3], title="HER")

# plot_all_fig(prefix=underarm_prefix, tag=tag_5)
# plot_all_fig(prefix=underarm_prefix, tag=tag_6)
# plot_all_fig(prefix=underarm_prefix, tag=tag_27)
# plot_all_fig(prefix=underarm_prefix, tag=tag_28)
# plot_all_fig(prefix=underarm_prefix, tag=tag_4)
# plot_all_fig(prefix=underarm_prefix, tag=tag_5)
# plot_all_fig(prefix=underarm_prefix, tag=tag_6)
# plot_all_fig(prefix=underarm_prefix, tag=tag_8)
# plot_all_fig(prefix=underarm_prefix, tag=tag_9)
# plot_all_fig(prefix=underarm_prefix, tag=tag_10)
# plot_all_fig(prefix=underarm_prefix, tag=tag_12)
# plot_all_fig(prefix=underarm_prefix, tag=tag_13)
# plot_all_fig(prefix=underarm_prefix, tag=tag_15)
# plot_all_fig(prefix=underarm_prefix, tag=tag_16)


# plot_all_fig(prefix=overarm_prefix, tag=tag_5)
# plot_all_fig(prefix=overarm_prefix, tag=tag_6)
# plot_all_fig(prefix=overarm_prefix, tag=tag_3)
# plot_all_fig(prefix=overarm_prefix, tag=tag_6)
# plot_all_fig(prefix=overarm_prefix, tag=tag_8)
# plot_all_fig(prefix=overarm_prefix, tag=tag_14)
# plot_all_fig(prefix=overarm_prefix, tag=tag_15)
# plot_all_fig(prefix=overarm_prefix, tag=tag_16)
# plot_all_fig(prefix=overarm_prefix, tag=tag_7)
# plot_all_fig(prefix=overarm_prefix, tag=tag_11)

