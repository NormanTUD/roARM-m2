import time
import threading

import numpy as np


class SimulatedArm:
    """Simulates the RoArm when no physical robot is connected.

    Provides the same interface as RoArmConnection but moves
    joints virtually with realistic timing.
    """

    def __init__(self):
        self._position = {
            "b": 0.0,
            "s": 0.0,
            "e": 90.0,
            "h": 180.0,
        }
        self._target = None
        self._torque_on = True
        self._gripper_open = True
        self._move_speed = 30.0
        self._moving = False
        self._lock = threading.Lock()

    def read_position_deg(self) -> dict:
        with self._lock:
            return self._position.copy()

    def read_position_averaged(self, n: int = 10, interval: float = 0.05) -> dict:
        pos = self.read_position_deg()
        for j in ["b", "s", "e", "h"]:
            pos[j] += np.random.normal(0, 0.02)
        return pos

    def move_to(self, b: float, s: float, e: float, h: float,
                spd: int = 20, acc: int = 10):
        with self._lock:
            self._target = {"b": b, "s": s, "e": e, "h": h}
            self._move_speed = spd * 1.5

    def move_to_fast(self, b: float, s: float, e: float, h: float,
                     spd: int = 50, acc: int = 30):
        with self._lock:
            self._target = {"b": b, "s": s, "e": e, "h": h}
            self._move_speed = spd * 2.0

    def step_simulation(self, dt: float):
        with self._lock:
            if self._target is None:
                return
            if not self._torque_on:
                return

            all_arrived = True
            for j in ["b", "s", "e", "h"]:
                diff = self._target[j] - self._position[j]
                if abs(diff) < 0.01:
                    self._position[j] = self._target[j]
                else:
                    all_arrived = False
                    max_step = self._move_speed * dt
                    step = max(-max_step, min(max_step, diff))
                    self._position[j] += step

            if all_arrived:
                self._target = None
                self._moving = False
            else:
                self._moving = True

    def wait_until_settled(self, tolerance_deg: float = 0.2,
                           stable_count: int = 6):
        time.sleep(0.3)

    def torque_on(self):
        with self._lock:
            self._torque_on = True

    def torque_off(self):
        with self._lock:
            self._torque_on = False
            self._target = None

    def gripper_open(self):
        self._gripper_open = True

    def gripper_close(self):
        self._gripper_open = False

    def close(self):
        pass

    @property
    def is_simulated(self) -> bool:
        return True

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._target is not None
