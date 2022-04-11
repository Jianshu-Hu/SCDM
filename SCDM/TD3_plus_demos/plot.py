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

pen_prefix = "TD3_PenSpin-v0_"

# overarm and underarm
tag_1 = ["1_random_goal_demo", "2_random_goal_demo", "3_random_goal_demo", "4_random_goal_demo", "5_random_goal_demo"]
# tag_1 = ["1_random_goal_demo", "2_random_goal_demo", "3_random_goal_demo"]
tag_2 = ["1_random_goal_demo_with_normaliser", "2_random_goal_demo_with_normaliser", "3_random_goal_demo_with_normaliser"]

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

tag_77 = ["1_random_goal_demo_add_artificial_transitions_restricted_02_actions",
          "2_random_goal_demo_add_artificial_transitions_restricted_02_actions",
          "3_random_goal_demo_add_artificial_transitions_restricted_02_actions"]

tag_78 = ["1_random_goal_demo_add_artificial_transitions_random_one_hand",
          "2_random_goal_demo_add_artificial_transitions_random_one_hand",
          "3_random_goal_demo_add_artificial_transitions_random_one_hand"]
tag_79 = ["1_random_goal_demo_policy_freq_3",
          "2_random_goal_demo_policy_freq_3",
          "3_random_goal_demo_policy_freq_3"]

tag_80 = ["1_random_goal_demo_add_artificial_transitions_forward_one_step",
          "2_random_goal_demo_add_artificial_transitions_forward_one_step",
          "3_random_goal_demo_add_artificial_transitions_forward_one_step"]

tag_81 = ["1_random_goal_demo_add_artificial_transitions_decay_Q_loss",
          "2_random_goal_demo_add_artificial_transitions_decay_Q_loss",
          "3_random_goal_demo_add_artificial_transitions_decay_Q_loss"]

tag_82 = ["1_random_goal_demo_add_artificial_transitions_forward_one_step_for_target_evaluation",
          "2_random_goal_demo_add_artificial_transitions_forward_one_step_for_target_evaluation",
          "3_random_goal_demo_add_artificial_transitions_forward_one_step_for_target_evaluation"]

tag_83 = ["1_random_goal_demo_add_artificial_transitions_epsilon_greedy",
          "2_random_goal_demo_add_artificial_transitions_epsilon_greedy",
          "3_random_goal_demo_add_artificial_transitions_epsilon_greedy"]

tag_84 = ["1_random_goal_demo_fix_target_rotation",
          "2_random_goal_demo_fix_target_rotation",
          "3_random_goal_demo_fix_target_rotation"]

tag_85 = ["1_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q",
          "2_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q",
          "3_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q"]

tag_86 = ["1_random_goal_demo_add_artificial_transitions_policy_action",
          "2_random_goal_demo_add_artificial_transitions_policy_action",
          "3_random_goal_demo_add_artificial_transitions_policy_action"]

tag_87 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_1",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_1"]
tag_88 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_3",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_3",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_3"]

tag_91 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action"]

tag_92 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_0",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_0",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_0"]
tag_93 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000"]

tag_94 = ["1_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_after_3e6",
          "2_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_after_3e6",
          "3_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_after_3e6"]

tag_95 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_one_step_return_for_real_transition",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_one_step_return_for_real_transition",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_one_step_return_for_real_transition"]

tag_96 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_debug_loss_calculation",
          "2_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_debug_loss_calculation",
          "3_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000_debug_loss_calculation"]

tag_97 = ["1_random_goal_demo_add_artificial_transitions_average_loss",
          "2_random_goal_demo_add_artificial_transitions_average_loss",
          "3_random_goal_demo_add_artificial_transitions_average_loss"]

tag_98 = ["1_random_goal_demo_policy_freq_4",
          "2_random_goal_demo_policy_freq_4",
          "3_random_goal_demo_policy_freq_4"]

tag_99 = ["1_random_goal_demo_policy_freq_5",
          "2_random_goal_demo_policy_freq_5",
          "3_random_goal_demo_policy_freq_5"]

tag_100 = ["1_random_goal_demo_add_artificial_transitions_favor_larger_diff",
           "2_random_goal_demo_add_artificial_transitions_favor_larger_diff",
           "3_random_goal_demo_add_artificial_transitions_favor_larger_diff"]

tag_101 = ["1_random_goal_demo_add_artificial_transitions_favor_smaller_diff",
           "2_random_goal_demo_add_artificial_transitions_favor_smaller_diff",
           "3_random_goal_demo_add_artificial_transitions_favor_smaller_diff"]

tag_102 = ["1_random_goal_demo_add_artificial_transitions_favor_higher_target_Q",
           "2_random_goal_demo_add_artificial_transitions_favor_higher_target_Q",
           "3_random_goal_demo_add_artificial_transitions_favor_higher_target_Q"]

tag_103 = ["1_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_when_having_a_small_diff",
           "2_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_when_having_a_small_diff",
           "3_random_goal_demo_add_artificial_transitions_filter_with_higher_target_Q_when_having_a_small_diff"]

tag_104 = ["1_random_goal_demo_add_artificial_transitions_filter_with_small_model_error",
           "2_random_goal_demo_add_artificial_transitions_filter_with_small_model_error",
           "3_random_goal_demo_add_artificial_transitions_filter_with_small_model_error"]

tag_105 = ["1_random_goal_demo_add_artificial_transitions_before_3e6",
           "2_random_goal_demo_add_artificial_transitions_before_3e6",
           "3_random_goal_demo_add_artificial_transitions_before_3e6"]

tag_106 = ["1_random_goal_demo_add_artificial_transitions_policy_freq_3",
           "2_random_goal_demo_add_artificial_transitions_policy_freq_3",
           "3_random_goal_demo_add_artificial_transitions_policy_freq_3"]

tag_107 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action_true_loss_for_true_transition",
           "2_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action_true_loss_for_true_transition",
           "3_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action_true_loss_for_true_transition"]

tag_108 = ["1_random_goal_demo_add_artificial_transitions_invariance_forward_H_10",
           "2_random_goal_demo_add_artificial_transitions_invariance_forward_H_10",
           "3_random_goal_demo_add_artificial_transitions_invariance_forward_H_10"]

tag_109 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_true_loss_for_true_transition",
           "2_random_goal_demo_add_artificial_transitions_MVE_H_1_true_loss_for_true_transition",
           "3_random_goal_demo_add_artificial_transitions_MVE_H_1_true_loss_for_true_transition"]

tag_110 = ["1_random_goal_demo_add_artificial_transitions_invariance_forward_H_20",
           "2_random_goal_demo_add_artificial_transitions_invariance_forward_H_20",
           "3_random_goal_demo_add_artificial_transitions_invariance_forward_H_20"]

tag_111 = ["1_random_goal_demo_add_artificial_transitions_with_normaliser",
           "2_random_goal_demo_add_artificial_transitions_with_normaliser",
           "3_random_goal_demo_add_artificial_transitions_with_normaliser"]

tag_112 = ["1_random_goal_demo_add_artificial_transitions_quadratic_pdf",
           "2_random_goal_demo_add_artificial_transitions_quadratic_pdf",
           "3_random_goal_demo_add_artificial_transitions_quadratic_pdf"]

tag_113 = ["1_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q",
           "2_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q",
           "3_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q"]

tag_114 = ["1_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q_with_normaliser",
           "2_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q_with_normaliser",
           "3_random_goal_demo_add_artificial_transitions_decay_filter_with_higher_target_Q_with_normaliser"]

tag_115 = ["1_random_goal_demo_MVE_H_1_after_1e6",
           "2_random_goal_demo_MVE_H_1_after_1e6",
           "3_random_goal_demo_MVE_H_1_after_1e6"]
tag_116 = ["1_random_goal_demo_MVE_H_10_after_1e6",
           "2_random_goal_demo_MVE_H_10_after_1e6",
           "3_random_goal_demo_MVE_H_10_after_1e6"]
tag_117 = ["1_random_goal_demo_MVE_H_5_after_1e6",
           "2_random_goal_demo_MVE_H_5_after_1e6",
           "3_random_goal_demo_MVE_H_5_after_1e6"]

tag_118 = ["1_random_goal_demo_add_artificial_transitions_invariance_forward_H_20_with_normaliser",
           "2_random_goal_demo_add_artificial_transitions_invariance_forward_H_20_with_normaliser",
           "3_random_goal_demo_add_artificial_transitions_invariance_forward_H_20_with_normaliser"]
tag_119 = ["1_random_goal_demo_MVE_H_1_with_normaliser",
           "2_random_goal_demo_MVE_H_1_with_normaliser",
           "3_random_goal_demo_MVE_H_1_with_normaliser"]
tag_120 = ["1_random_goal_demo_add_artificial_transitions_filter_with_model_error_estimated_by_target_Q_error",
           "2_random_goal_demo_add_artificial_transitions_filter_with_model_error_estimated_by_target_Q_error",
           "3_random_goal_demo_add_artificial_transitions_filter_with_model_error_estimated_by_target_Q_error"]

tag_121 = ["1_random_goal_demo_enable_exploratory_policy",
           "2_random_goal_demo_enable_exploratory_policy",
           "3_random_goal_demo_enable_exploratory_policy"]

tag_122 = ["1_random_goal_demo_enable_exploratory_policy_exploration",
           "2_random_goal_demo_enable_exploratory_policy_exploration",
           "3_random_goal_demo_enable_exploratory_policy_exploration"]

tag_123 = ["1_random_goal_demo_add_artificial_transitions_initialize_with_10_bias_in_final_layer_of_Q",
           "2_random_goal_demo_add_artificial_transitions_initialize_with_10_bias_in_final_layer_of_Q",
           "3_random_goal_demo_add_artificial_transitions_initialize_with_10_bias_in_final_layer_of_Q"]

tag_124 = ["1_random_goal_demo_add_artificial_transitions_before_3e6_with_normaliser",
           "2_random_goal_demo_add_artificial_transitions_before_3e6_with_normaliser",
           "3_random_goal_demo_add_artificial_transitions_before_3e6_with_normaliser"]

tag_125 = ["1_add_artificial_transitions_filter_with_higher_target_Q_after_3e6_with_normaliser",
           "2_add_artificial_transitions_filter_with_higher_target_Q_after_3e6_with_normaliser",
           "3_add_artificial_transitions_filter_with_higher_target_Q_after_3e6_with_normaliser"]

tag_126 = ["1_add_artificial_transitions_filter_with_target_Q_error_after_3e6_with_normaliser",
           "2_add_artificial_transitions_filter_with_target_Q_error_after_3e6_with_normaliser",
           "3_add_artificial_transitions_filter_with_target_Q_error_after_3e6_with_normaliser"]

tag_127 = ["1_add_artificial_transitions_filter_with_model_error_after_3e6_with_normaliser",
           "2_add_artificial_transitions_filter_with_model_error_after_3e6_with_normaliser",
           "3_add_artificial_transitions_filter_with_model_error_after_3e6_with_normaliser"]
tag_128 = ["1_add_artificial_transitions_with_normaliser",
           "2_add_artificial_transitions_with_normaliser",
           "3_add_artificial_transitions_with_normaliser"]

tag_129 = ["1_add_artificial_transitions_policy_action",
           "2_add_artificial_transitions_policy_action",
           "3_add_artificial_transitions_policy_action"]

tag_130 = ["1_add_artificial_transitions_selecting_among_3_actions_with_highest_target_Q",
           "2_add_artificial_transitions_selecting_among_3_actions_with_highest_target_Q",
           "3_add_artificial_transitions_selecting_among_3_actions_with_highest_target_Q"]

tag_131 = ["1_add_artificial_transitions_decaying_Q_loss_with_normaliser",
           "2_add_artificial_transitions_decaying_Q_loss_with_normaliser",
           "3_add_artificial_transitions_decaying_Q_loss_with_normaliser"]

tag_132 = ["1_with_normaliser",
           "2_with_normaliser",
           "3_with_normaliser"]

tag_133 = ["1_add_artificial_transitions_epsilon_decay_with_normaliser",
           "2_add_artificial_transitions_epsilon_decay_with_normaliser",
           "3_add_artificial_transitions_epsilon_decay_with_normaliser"]

tag_134 = ["1_add_artificial_transitions_epsilon_greedy_with_normaliser",
           "2_add_artificial_transitions_epsilon_greedy_with_normaliser",
           "3_add_artificial_transitions_epsilon_greedy_with_normaliser"]

tag_135 = ["1_add_artificial_transitions_selecting_among_2_actions_with_highest_target_Q",
           "2_add_artificial_transitions_selecting_among_2_actions_with_highest_target_Q",
           "3_add_artificial_transitions_selecting_among_2_actions_with_highest_target_Q"]

tag_136 = ["1_MVE_H_1_one_step_return_for_true_transition",
           "2_MVE_H_1_one_step_return_for_true_transition",
           "3_MVE_H_1_one_step_return_for_true_transition"]

tag_137 = ["1_add_artificial_transitions_forward_1_more_steps",
           "2_add_artificial_transitions_forward_1_more_steps",
           "3_add_artificial_transitions_forward_1_more_steps"]

tag_138 = ["1_add_artificial_transitions_epsilon_decay_slower_7_with_normaliser",
           "2_add_artificial_transitions_epsilon_decay_slower_7_with_normaliser",
           "3_add_artificial_transitions_epsilon_decay_slower_7_with_normaliser"]

tag_139 = ["1_apply_model_in_policy_gradient_with_normaliser",
           "2_apply_model_in_policy_gradient_with_normaliser",
           "3_apply_model_in_policy_gradient_with_normaliser"]

tag_140 = ["1_apply_model_in_policy_gradient_unbiased_with_normaliser",
           "2_apply_model_in_policy_gradient_unbiased_with_normaliser",
           "3_apply_model_in_policy_gradient_unbiased_with_normaliser"]

tag_141 = ["1_apply_model_in_policy_gradient_only_through_dynamics",
           "2_apply_model_in_policy_gradient_only_through_dynamics",
           "3_apply_model_in_policy_gradient_only_through_dynamics"]

tag_142 = ["1_apply_model_in_policy_gradient_only_through_new_action",
           "2_apply_model_in_policy_gradient_only_through_new_action",
           "3_apply_model_in_policy_gradient_only_through_new_action"]

tag_143 = ["1_epsilon_greedy_forward_one_step_for_policy_action",
           "2_epsilon_greedy_forward_one_step_for_policy_action",
           "3_epsilon_greedy_forward_one_step_for_policy_action"]

tag_144 = ["1_MVE_H_1_without_noise",
           "2_MVE_H_1_without_noise",
           "3_MVE_H_1_without_noise"]

tag_145 = ["1_add_artificial_transtions_policy_action_decaying_clipped_noise",
           "2_add_artificial_transtions_policy_action_decaying_clipped_noise",
           "3_add_artificial_transtions_policy_action_decaying_clipped_noise"]

tag_146 = ["1_MVE_H_1_after_3e6",
           "2_MVE_H_1_after_3e6",
           "3_MVE_H_1_after_3e6"]

tag_147 = ["1_epsilon_greedy_forward_one_step_for_policy_action_correct",
           "2_epsilon_greedy_forward_one_step_for_policy_action_correct",
           "3_epsilon_greedy_forward_one_step_for_policy_action_correct"]

tag_148 = ["1_with_normaliser",
           "2_with_normaliser",
           "3_with_normaliser"]

tag_149 = ["1_epsilon_greedy_policy_action_MVE_H_1",
           "2_epsilon_greedy_policy_action_MVE_H_1",
           "3_epsilon_greedy_policy_action_MVE_H_1"]

tag_150 = ["1_with_normaliser_without_high_initialization_critic",
           "2_with_normaliser_without_high_initialization_critic",
           "3_with_normaliser_without_high_initialization_critic"]

tag_151 = ["1_policy_action_with_decaying_clipped_gaussian_noise",
           "2_policy_action_with_decaying_clipped_gaussian_noise",
           "3_policy_action_with_decaying_clipped_gaussian_noise"]

tag_152 = ["1_policy_action_with_decaying_clipped_uniform_noise",
           "2_policy_action_with_decaying_clipped_uniform_noise",
           "3_policy_action_with_decaying_clipped_uniform_noise"]

tag_153 = ["1_policy_action_with_decaying_clipped_05_gaussian_noise",
           "2_policy_action_with_decaying_clipped_05_gaussian_noise",
           "3_policy_action_with_decaying_clipped_05_gaussian_noise"]

tag_154 = ["1_policy_action_with_decaying_clipped_05_uniform_noise",
           "2_policy_action_with_decaying_clipped_05_uniform_noise",
           "3_policy_action_with_decaying_clipped_05_uniform_noise"]

tag_155 = ["1_policy_action_with_decaying_variance_gaussian_noise",
           "2_policy_action_with_decaying_variance_gaussian_noise",
           "3_policy_action_with_decaying_variance_gaussian_noise"]

tag_156 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_high_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_high_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_high_initialization"]

tag_157 = ["1_policy_action_with_decaying_smaller_variance_gaussian_noise",
           "2_policy_action_with_decaying_smaller_variance_gaussian_noise",
           "3_policy_action_with_decaying_smaller_variance_gaussian_noise"]

tag_158 = ["1_policy_action_with_decaying_variance_gaussian_noise_with_high_initialization",
           "2_policy_action_with_decaying_variance_gaussian_noise_with_high_initialization",
           "3_policy_action_with_decaying_variance_gaussian_noise_with_high_initialization"]

tag_159 = ["1_policy_action_with_decaying_clipped_uniform_noise_with_high_initialization",
           "2_policy_action_with_decaying_clipped_uniform_noise_with_high_initialization",
           "3_policy_action_with_decaying_clipped_uniform_noise_with_high_initialization"]

tag_160 = ["1_policy_action_with_decaying_clipped_gaussian_noise_filter_with_variance_with_high_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_filter_with_variance_with_high_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_filter_with_variance_with_high_initialization"]

tag_161 = ["1_policy_action_with_scheduled_decaying_clipped_gaussian_noise_with_high_initialization",
           "2_policy_action_with_scheduled_decaying_clipped_gaussian_noise_with_high_initialization",
           "3_policy_action_with_scheduled_decaying_clipped_gaussian_noise_with_high_initialization"]

tag_162 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_25_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_25_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_25_initialization"]

tag_163 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_35_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_35_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_35_initialization"]

tag_164 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_45_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_45_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_45_initialization"]

tag_165 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_1000_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_1000_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_1000_initialization"]

tag_166 = ["1_decaying_clipped_gaussian_noise_filter_with_variance_divided_by_max_with_high_initialization",
           "2_decaying_clipped_gaussian_noise_filter_with_variance_divided_by_max_with_high_initialization",
           "3_decaying_clipped_gaussian_noise_filter_with_variance_divided_by_max_with_high_initialization"]

tag_167 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_500_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_500_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_500_initialization"]

tag_168 = ["1_decaying_clipped_gaussian_noise_normalize_with_max_variance_so_far_with_high_initialization",
           "2_decaying_clipped_gaussian_noise_normalize_with_max_variance_so_far_with_high_initialization",
           "3_decaying_clipped_gaussian_noise_normalize_with_max_variance_so_far_with_high_initialization"]

tag_168 = ["1_decaying_clipped_gaussian_noise_normalize_with_max_difference_so_far_with_high_initialization",
           "2_decaying_clipped_gaussian_noise_normalize_with_max_difference_so_far_with_high_initialization",
           "3_decaying_clipped_gaussian_noise_normalize_with_max_difference_so_far_with_high_initialization"]

tag_169 = ["1_decaying_clipped_gaussian_noise_normalize_with_max_difference_with_high_initialization",
           "2_decaying_clipped_gaussian_noise_normalize_with_max_difference_with_high_initialization",
           "3_decaying_clipped_gaussian_noise_normalize_with_max_difference_with_high_initialization"]

tag_170 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_20_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_20_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_20_initialization"]

tag_171 = ["1_policy_action_with_decaying_clipped_gaussian_noise_with_200_initialization",
           "2_policy_action_with_decaying_clipped_gaussian_noise_with_200_initialization",
           "3_policy_action_with_decaying_clipped_gaussian_noise_with_200_initialization"]

tag_172 = ["1_scheduled_decaying_target_from_current_policy_with_high_initialization",
           "2_scheduled_decaying_target_from_current_policy_with_high_initialization",
           "3_scheduled_decaying_target_from_current_policy_with_high_initialization"]


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
    fig, axs = plt.subplots(2, 1)
    for i in range(len(tag)):
        loss = np.load('results/'+prefix + tag[i] + "_transition_model_loss.npy")
        axs[0].plot(range(len(loss)), loss, label=tag[i]+"_transition_model_loss")
        axs[0].set_xlabel('timesteps')
        axs[0].set_ylabel('model loss')
        axs[0].legend()

        loss = np.load('results/'+prefix + tag[i] + "_reward_model_loss.npy")
        axs[1].plot(range(len(loss)), loss, label=tag[i]+"_reward_model_loss")
        axs[1].set_xlabel('timesteps')
        axs[1].set_ylabel('model loss')
        axs[1].legend()

        axs[0].set_title(prefix + tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_transition_loss')

def plot_variance(prefix=overarm_prefix, tag=tag_59, plot_or_save='save'):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(tag)):
        loss = np.load('results/'+prefix + tag[i] + "_variance.npy")
        averaged_loss = np.zeros((int(loss.shape[0]/1000)))
        for j in range(averaged_loss.shape[0]):
            averaged_loss[j] = np.sum(loss[j*1000:(j+1)*1000])/1000

        # axs.plot(range(len(loss)), loss, label=tag[i] + "_variance")
        # axs.set_xlabel('timesteps')
        # axs.set_ylabel('variance')
        axs.plot(range(len(averaged_loss)), averaged_loss, label=tag[i]+"_variance")
        axs.set_xlabel('timesteps/1000')
        axs.set_ylabel('averaged variance over 1000 timesteps')

        axs.legend()

        axs.set_title(prefix + tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_variance')

def plot_debug_value(prefix=overarm_prefix, tag=tag_59, plot_or_save='save'):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(tag)):
        loss = np.load('results/'+prefix + tag[i] + "_debug_value.npy")
        averaged_loss = np.zeros((int(loss.shape[0]/1000)))
        for j in range(averaged_loss.shape[0]):
            averaged_loss[j] = np.sum(loss[j*1000:(j+1)*1000])/1000
        axs.plot(range(len(averaged_loss)), averaged_loss, label=tag[i]+"_debug_value")
        axs.set_xlabel('timesteps/1000')
        axs.set_ylabel('averaged debug value over 1000 timesteps')
        axs.legend()

        axs.set_title(prefix + tag[0])
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/' + prefix + tag[0] + '_debug_value')

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
#plot_all_fig(prefix=overarm_prefix, tag=tag_57)
#plot_all_fig(prefix=overarm_prefix, tag=tag_59)
#plot_all_fig(prefix=overarm_prefix, tag=tag_60)
#plot_all_fig(prefix=overarm_prefix, tag=tag_61)
#plot_all_fig(prefix=overarm_prefix, tag=tag_62)
#plot_all_fig(prefix=underarm_prefix, tag=tag_67)
# plot_all_fig(prefix=underarm_prefix, tag=tag_125)
# plot_transition_model_loss(prefix=underarm_prefix, tag=tag_111)
# plot_all_fig(prefix=overarm_prefix, tag=tag_92)
# plot_transition_model_loss(prefix=overarm_prefix, tag=tag_111)
# plot_all_fig(prefix=underarmhard_prefix, tag=tag_71)
# plot_actor_critic_loss(overarm_prefix, tag_1)
# plot_actor_critic_loss(underarm_prefix, tag_1)
# plot_actor_critic_loss(overarm_prefix, tag_151)
# plot_actor_critic_loss(overarm_prefix, tag_152)
# compare_policy_critic(overarm_prefix, overarm_prefix, tag_76)
# compare_policy_critic(overarm_prefix, underarm_prefix, tag_80)
# compare_policy_critic(overarm_prefix, underarm_prefix, tag_92)
# compare_policy_critic(overarm_prefix, underarm_prefix, tag_85)

# compare_policy_critic(underarm_prefix, overarm_prefix, tag_76)
# compare_policy_critic(underarm_prefix, underarm_prefix, tag_80)
# compare_policy_critic(underarm_prefix, underarm_prefix, tag_92)
# compare_policy_critic(underarm_prefix, underarm_prefix, tag_85)
# plot_transition_model_loss(prefix=overarm_prefix, tag=tag_151)
# plot_transition_model_loss(prefix=overarm_prefix, tag=tag_152)

# plot_all_fig(overarm_prefix, tag=tag_161)

plot_debug_value(overarm_prefix, tag=tag_172)
plot_debug_value(underarm_prefix, tag=tag_172)

plot_variance(overarm_prefix, tag=tag_168)
plot_variance(underarm_prefix, tag=tag_168)

# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_79, tag_98, tag_99], title="policy_freq")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_79, tag_98, tag_99], title="policy_freq")

# tag_test = ["0_test","0_test_exploration"]
# tag_expl = ["0_test_exploration"]
# compare_policy_critic(overarm_prefix,overarm_prefix,tag_test,plot_or_save='plot')
# plot_actor_critic_loss(overarm_prefix,tag_expl,plot_or_save='plot')
# plot_transition_model_loss(prefix=pen_prefix, tag=tag_132)

# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_79, tag_106], title="policy_freq_3")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_79, tag_106], title="policy_freq_3")
# compare(prefix=pen_prefix,tag_list=[tag_132, tag_128, tag_131, tag_134, tag_138, tag_143, tag_147], title="add_transitions")
# compare(prefix=pen_prefix,tag_list=[tag_132, tag_151, tag_152], title="noise_policy_action")
compare(prefix=pen_prefix, tag_list=[tag_132, tag_156, tag_158, tag_159], title="noise_policy_action_with_high_initialization")
compare(prefix=pen_prefix, tag_list=[tag_132, tag_156, tag_165, tag_167, tag_171], title="high_initialization")

#compare(prefix=overarm_prefix, tag_list=[tag_79, tag_76, tag_77, tag_81, tag_97, tag_105, tag_112], title="add_transitions")
#compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_114, tag_118, tag_119, tag_120, tag_121, tag_123], title="with_normaliser")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_125, tag_126, tag_127], title="filter")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_124, tag_129, tag_131, tag_136, tag_137], title="different_action_with_normaliser")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_133, tag_143, tag_147, tag_149], title="epsilon_greedy")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_145, tag_151, tag_152, tag_153, tag_154, tag_155, tag_157], title="noise_policy_action")
# tag_158_temp = ["1_policy_action_with_decaying_variance_gaussian_noise_with_high_initialization",
#            "3_policy_action_with_decaying_variance_gaussian_noise_with_high_initialization"]
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_156, tag_158_temp, tag_159], title="noise_policy_action_with_high_initialization")
compare(prefix=overarm_prefix, tag_list=[tag_2, tag_156, tag_160, tag_166, tag_168, tag_169], title="noise_policy_action_with_improvement")
compare(prefix=overarm_prefix, tag_list=[tag_2, tag_156, tag_161, tag_172], title="noise_policy_action_scheduled_decaying")
compare(prefix=overarm_prefix, tag_list=[tag_2, tag_156, tag_163, tag_164, tag_170], title="high_initialization")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_130, tag_135], title="selecting_action")
# compare(prefix=overarm_prefix, tag_list=[tag_2, tag_139, tag_140, tag_141, tag_142], title="model_in_policy_gradient")
compare(prefix=overarm_prefix, tag_list=[tag_2, tag_128, tag_144, tag_145, tag_146, tag_148,tag_150], title='play_with_noise')


#compare(prefix=overarm_prefix, tag_list=[tag_79, tag_76, tag_85, tag_94, tag_103, tag_104, tag_113], title="filter_transitions")
#compare(prefix=overarm_prefix, tag_list=[tag_79, tag_76, tag_78, tag_83, tag_86, tag_100, tag_101, tag_102], title="different_actions")
#compare(prefix=overarm_prefix, tag_list=[tag_79, tag_76, tag_108, tag_110], title="add_invariance_H")
#compare(prefix=overarm_prefix, tag_list=[tag_79, tag_80, tag_82], title="forward_one_step")
#compare(prefix=overarm_prefix, tag_list=[tag_1, tag_93, tag_109, tag_107, tag_115, tag_116, tag_117], title="MVE")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_84], title="fix_target_rotation")

#compare(prefix=underarm_prefix, tag_list=[tag_79, tag_76, tag_77, tag_81, tag_97, tag_105, tag_112], title="add_transitions")
#compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_114, tag_118, tag_119, tag_120, tag_121, tag_123], title="with_normaliser")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_125, tag_126, tag_127], title="filter")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_124, tag_129, tag_131, tag_136,tag_137], title="different_action_with_normaliser")
# tag_143 = ["1_epsilon_greedy_forward_one_step_for_policy_action",
#            "2_epsilon_greedy_forward_one_step_for_policy_action"]
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_133, tag_143, tag_147, tag_149], title="epsilon_greedy")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_145, tag_151, tag_152], title="noise_policy_action")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_156, tag_158, tag_159], title="noise_policy_action_with_high_initialization")
compare(prefix=underarm_prefix, tag_list=[tag_2, tag_156, tag_160, tag_166, tag_168, tag_169], title="noise_policy_action_with_improvement")
compare(prefix=underarm_prefix, tag_list=[tag_2, tag_156, tag_161, tag_172], title="noise_policy_action_scheduled_decaying")
compare(prefix=underarm_prefix, tag_list=[tag_2, tag_156, tag_162, tag_163, tag_170], title="high_initialization")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_130, tag_135], title="selecting_action")
# compare(prefix=underarm_prefix, tag_list=[tag_2, tag_139, tag_140, tag_141, tag_142], title="model_in_policy_gradient")
compare(prefix=underarm_prefix, tag_list=[tag_2, tag_128, tag_144, tag_145, tag_146, tag_148, tag_150], title='play_with_noise')

#compare(prefix=underarm_prefix, tag_list=[tag_79, tag_76, tag_85, tag_94, tag_103, tag_104, tag_113], title="filter_transitions")
#compare(prefix=underarm_prefix, tag_list=[tag_79, tag_76, tag_78, tag_83, tag_86, tag_100, tag_101, tag_102], title="different_actions")
#compare(prefix=underarm_prefix, tag_list=[tag_79, tag_76, tag_108, tag_110], title="add_invariance_H")
#compare(prefix=underarm_prefix, tag_list=[tag_79, tag_80, tag_82], title="forward_one_step")
# tag_107 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action_true_loss_for_true_transition",
#            "3_random_goal_demo_add_artificial_transitions_MVE_H_1_random_action_true_loss_for_true_transition"]
# tag_93 = ["1_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000",
#           "3_random_goal_demo_add_artificial_transitions_MVE_H_1_train_model_for_25000"]
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_93, tag_109, tag_107, tag_115, tag_116,tag_117], title="MVE")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_84], title="fix_target_rotation")

#compare(prefix=underarmhard_prefix, tag_list=[tag_71, tag_76, tag_84], title="add_transitions")

# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_57, tag_63, tag_64, tag_65], title="tuned_hand_invariance")
# compare(prefix=underarm_prefix, tag_list=[tag_57, tag_73, tag_74], title="auto_hand_invariance")

# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_57, tag_70, tag_65, tag_66, tag_72], title="tuned_hand_invariance")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_57, tag_73, tag_74], title="auto_hand_invariance")
# compare(prefix=underarm_prefix, tag_list=[tag_57, tag_67, tag_68], title="demo_len")
# plot_all_fig(prefix=overarm_prefix, tag=tag_69)
# compare_policy_critic(overarm_prefix, overarm_prefix, tag_69)

# compare(prefix=underarmhard_prefix, tag_list=[tag_71], title="baseline")
# compare(prefix=underarm_prefix, tag_list=[tag_1, tag_21, tag_43, tag_24, tag_28], title="baseline")
# compare(prefix=underarm_prefix, tag_list=[tag_21, tag_51, tag_52, tag_53, tag_54], title="new_hand_invariance")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_22, tag_23, tag_24, tag_25], title="baseline")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_45, tag_21, tag_24, tag_43, tag_44], title="baseline")
# compare(prefix=overarm_prefix, tag_list=[tag_21, tag_48, tag_49, tag_50, tag_55, tag_56, tag_59], title="decay rate")
# compare(prefix=overarm_prefix, tag_list=[tag_1, tag_21, tag_57, tag_58, tag_60, tag_61, tag_62], title="exclude demo")
# compare(prefix=overarm_prefix, tag_list=[tag_21, tag_41, tag_42, tag_46, tag_47], title="hand_invariance")
# compare(prefix=overarm_prefix, tag_list=[tag_21, tag_51, tag_52, tag_53,tag_54], title="new_hand_invariance")
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

