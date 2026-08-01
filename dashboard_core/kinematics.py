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

JOINTS = ["b", "s", "e"]

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

# ============================================================
# WAYPOINT CAPTURE / PLAYBACK
# ============================================================

# How long torque is enabled for the gravity-compensated position read
# when the user presses Space in waypoint capture mode.
WAYPOINT_TORQUE_PULSE_S = 0.25

# Settle time after torque-on before reading the position.
WAYPOINT_CAPTURE_SETTLE_S = 0.10

# Default trapezoidal motion profile limits for waypoint playback.
# Conservative defaults: smooth and gentle motion suitable for putting the
# gripper near fragile objects without overshoot. Increase via the speed
# multiplier in the play tab if you need to go faster.
WAYPOINT_V_MAX_DEG_S = 40.0
WAYPOINT_A_MAX_DEG_S2 = 90.0

# Servo-side speed/acceleration used during waypoint streaming.
# Much lower than STREAM_SPD/STREAM_ACC (50/30) so the servo doesn't
# apply its own aggressive profile on top of our trapezoid target.
WAYPOINT_STREAM_SPD = 15
WAYPOINT_STREAM_ACC = 8

# Pause at each captured waypoint so the arm has time to settle before
# the next motion command stream starts.
WAYPOINT_SETTLE_S = 0.5

# User-settable playback speed (multiplier on the default v_max).
WAYPOINT_DEFAULT_SPEED = 0.7

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
