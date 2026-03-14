"""Task Variable definitions and Jacobian construction for UCM analysis."""

import numpy as np


def build_jacobian_tv1(n_steps=8, action_dim=7):
    """
    TV1: Cumulative EE Displacement = [sum(dx), sum(dy), sum(dz)]
    J is (3, 56): sums position deltas across all chunk steps.
    Null space (53D) = orientation, gripper, and redundant position DOFs.
    """
    D = n_steps * action_dim
    J = np.zeros((3, D))
    for step in range(n_steps):
        base = step * action_dim
        J[0, base + 0] = 1.0
        J[1, base + 1] = 1.0
        J[2, base + 2] = 1.0
    return J


def build_jacobian_tv2(n_steps=8, action_dim=7):
    """
    TV2: Cumulative Position + Orientation (6D)
    J is (6, 56): sums pos+orient deltas across all chunk steps.
    Null space (50D) = gripper + redundant pose DOFs.
    """
    D = n_steps * action_dim
    J = np.zeros((6, D))
    for step in range(n_steps):
        base = step * action_dim
        for dim in range(6):
            J[dim, base + dim] = 1.0
    return J


def build_jacobian_tv3(n_steps=8, action_dim=7):
    """
    TV3: All per-step positions (24D)
    J is (24, 56): identity for position dims, zero for orientation/gripper.
    Null space (32D) = orientation + gripper at each step.
    """
    D = n_steps * action_dim
    tv_dim = n_steps * 3
    J = np.zeros((tv_dim, D))
    for step in range(n_steps):
        base_action = step * action_dim
        base_tv = step * 3
        J[base_tv + 0, base_action + 0] = 1.0
        J[base_tv + 1, base_action + 1] = 1.0
        J[base_tv + 2, base_action + 2] = 1.0
    return J


def build_jacobian_tv1_no_gripper(n_steps=8, action_dim=6):
    """TV1 without gripper dimension. Action is 48D (6D x 8 steps)."""
    D = n_steps * action_dim
    J = np.zeros((3, D))
    for step in range(n_steps):
        base = step * action_dim
        J[0, base + 0] = 1.0
        J[1, base + 1] = 1.0
        J[2, base + 2] = 1.0
    return J


def build_jacobian_tv2_no_gripper(n_steps=8, action_dim=6):
    """TV2 without gripper dimension. Action is 48D."""
    D = n_steps * action_dim
    J = np.zeros((6, D))
    for step in range(n_steps):
        base = step * action_dim
        for dim in range(6):
            J[dim, base + dim] = 1.0
    return J


def build_jacobian_tv3_no_gripper(n_steps=8, action_dim=6):
    """TV3 without gripper dimension. Action is 48D. Task = 24D position."""
    D = n_steps * action_dim
    tv_dim = n_steps * 3
    J = np.zeros((tv_dim, D))
    for step in range(n_steps):
        base_action = step * action_dim
        base_tv = step * 3
        J[base_tv + 0, base_action + 0] = 1.0
        J[base_tv + 1, base_action + 1] = 1.0
        J[base_tv + 2, base_action + 2] = 1.0
    return J


TASK_VARIABLES = {
    "TV1_cumulative_position": build_jacobian_tv1,
    "TV2_cumulative_pos_orient": build_jacobian_tv2,
    "TV3_per_step_position": build_jacobian_tv3,
}

TASK_VARIABLES_NO_GRIPPER = {
    "TV1_cumulative_position": build_jacobian_tv1_no_gripper,
    "TV2_cumulative_pos_orient": build_jacobian_tv2_no_gripper,
    "TV3_per_step_position": build_jacobian_tv3_no_gripper,
}
