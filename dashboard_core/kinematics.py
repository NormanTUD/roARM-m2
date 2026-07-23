import math
from pathlib import Path

import numpy as np

# ============================================================
# KINEMATIK-KONSTANTEN
# ============================================================

BASE_HEIGHT = 75.0
UPPER_ARM = 206.0
FOREARM = 206.0
GRIPPER_LENGTH = 80.0

ENDPOINT_SPEEDS = [(8, 4), (5, 2), (3, 1)]
ENDPOINT_SETTLE_WAIT = 0.8
GRAVITY_COMP_SETTLE_MS = 30
MAX_TRACKING_ERROR = 15.0

MAX_DEG_PER_TICK = 1.5
LOOKAHEAD_MS = 40
LOOKAHEAD_S = 0.15

STREAM_SPD = 50
STREAM_ACC = 30

TRACKING_CHECK_INTERVAL_S = 2.0
TRACKING_CHECK_ENABLED = False

USE_BUSY_WAIT = True
MAX_SERIAL_BATCH = 3

STREAM_MIN_SEND_INTERVAL_S = 0.008
STREAM_FLUSH_INTERVAL = 200
STREAM_UI_UPDATE_INTERVAL_S = 1.0
STREAM_EVENT_PAUSE_S = 0.05

STREAM_PREBUFFER_COMMANDS = 5

# ============================================================
# ADAPTIVE TIMING CONSTANTS
# ============================================================

MIN_SPEED_FACTOR = 0.7
MAX_SPEED_FACTOR = 1.2
END_RAMP_PERCENT = 0.05
START_RAMP_PERCENT = 0.03

# ============================================================
# KONFIGURATION
# ============================================================

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

RECORD_HZ = 50
MOVE_THRESHOLD_DEG = 0.3
STREAM_HZ = 100

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def forward_kinematics(b_deg: float, s_deg: float, e_deg: float) -> dict:
    b_rad = math.radians(b_deg)
    s_rad = math.radians(90.0 - s_deg)

    base = np.array([0.0, 0.0, 0.0])

    elbow_local_x = UPPER_ARM * math.cos(s_rad)
    elbow_local_z = BASE_HEIGHT + UPPER_ARM * math.sin(s_rad)

    forearm_abs_angle = s_rad - math.radians(e_deg)

    total_forearm = FOREARM + GRIPPER_LENGTH
    gripper_local_x = elbow_local_x + total_forearm * math.cos(forearm_abs_angle)
    gripper_local_z = elbow_local_z + total_forearm * math.sin(forearm_abs_angle)

    cos_b = math.cos(b_rad)
    sin_b = math.sin(b_rad)

    def rotate_base(x, z):
        return np.array([x * cos_b, x * sin_b, z])

    return {
        "base": base,
        "shoulder": np.array([0.0, 0.0, BASE_HEIGHT]),
        "elbow": rotate_base(elbow_local_x, elbow_local_z),
        "gripper": rotate_base(gripper_local_x, gripper_local_z),
    }
