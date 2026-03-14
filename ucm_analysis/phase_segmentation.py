"""Episode phase segmentation for task-phase-specific UCM analysis."""

import numpy as np


def segment_episode_phases(expert_actions, gripper_threshold=0.5, motion_threshold=0.005):
    """
    Segment episode timesteps into manipulation phases.

    Args:
        expert_actions: (T, 7) expert actions [x,y,z,roll,pitch,yaw,gripper]
        gripper_threshold: threshold for gripper open/close
        motion_threshold: threshold for "large" position change

    Returns:
        labels: (T,) array of phase labels
            0=approach, 1=pre_grasp, 2=grasp, 3=transport, 4=place
    """
    T = len(expert_actions)
    labels = np.zeros(T, dtype=int)

    gripper = expert_actions[:, 6]
    pos_delta_norm = np.linalg.norm(expert_actions[:, :3], axis=1)

    gripper_open = gripper > gripper_threshold

    # Find gripper transitions
    gripper_close_idx = None
    gripper_open_idx = None
    for t in range(1, T):
        if gripper_open[t - 1] and not gripper_open[t] and gripper_close_idx is None:
            gripper_close_idx = t
        if not gripper_open[t - 1] and gripper_open[t] and gripper_close_idx is not None:
            gripper_open_idx = t
            break

    if gripper_close_idx is None:
        # No grasp detected — label everything as approach
        labels[:] = 0
        return labels

    # Approach: before grasp, large motion, gripper open
    # Pre-grasp: before grasp, small motion, gripper open
    for t in range(gripper_close_idx):
        if pos_delta_norm[t] > motion_threshold:
            labels[t] = 0  # approach
        else:
            labels[t] = 1  # pre_grasp

    # Grasp: gripper closing transition (small window)
    grasp_window = min(3, T - gripper_close_idx)
    labels[gripper_close_idx:gripper_close_idx + grasp_window] = 2

    if gripper_open_idx is not None:
        # Transport: gripper closed, moving
        labels[gripper_close_idx + grasp_window:gripper_open_idx] = 3
        # Place: gripper opening
        labels[gripper_open_idx:] = 4
    else:
        # No place detected — everything after grasp is transport
        labels[gripper_close_idx + grasp_window:] = 3

    return labels


PHASE_NAMES = {0: "approach", 1: "pre_grasp", 2: "grasp", 3: "transport", 4: "place"}
