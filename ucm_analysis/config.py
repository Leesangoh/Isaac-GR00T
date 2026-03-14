"""Configuration for UCM analysis."""

BRIDGE_DATA_DIR = "/mnt/md1/solee/data/bridge_lerobot"
VLA_ACTIONS_DIR = "/mnt/md1/solee/features/vla_actions"

N_STEPS = 8          # action chunk length
ACTION_DIM = 7       # [x, y, z, roll, pitch, yaw, gripper]
CONTINUOUS_DIMS = 6  # gripper excluded
ACTION_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
