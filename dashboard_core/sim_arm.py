import time
import threading
import math

import numpy as np


class SimulatedArm:
    """Simulates the RoArm when no physical robot is connected.

    Provides the same interface as RoArmConnection but moves
    joints virtually with realistic acceleration/deceleration curves.
    """

    def __init__(self):
        self._position = {
            "b": 0.0,
            "s": 0.0,
            "e": 90.0,
            "h": 180.0,
        }
        self._velocity = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        self._target = None
        self._torque_on = True
        self._gripper_open = True
        self._max_speed = 120.0
        self._accel = 300.0
        self._settled = True
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
            speed_scale = spd / 20.0
            self._max_speed = 120.0 * speed_scale
            self._accel = 300.0 * (acc / 10.0)
            self._settled = False

    def move_to_fast(self, b: float, s: float, e: float, h: float,
                     spd: int = 50, acc: int = 30):
        with self._lock:
            self._target = {"b": b, "s": s, "e": e, "h": h}
            speed_scale = spd / 20.0
            self._max_speed = 120.0 * speed_scale
            self._accel = 300.0 * (acc / 10.0)
            self._settled = False

    def step_simulation(self, dt: float):
        with self._lock:
            if self._target is None:
                return
            if not self._torque_on:
                return

            all_arrived = True
            for j in ["b", "s", "e", "h"]:
                diff = self._target[j] - self._position[j]
                dist = abs(diff)
                sign = 1.0 if diff > 0 else -1.0

                if dist < 0.005:
                    self._position[j] = self._target[j]
                    self._velocity[j] = 0.0
                    continue

                all_arrived = False

                decel_dist = (self._velocity[j] ** 2) / (2.0 * self._accel) if self._accel > 0 else 0

                if decel_dist >= dist * 0.9:
                    self._velocity[j] = max(0.0, self._velocity[j] - self._accel * dt)
                else:
                    self._velocity[j] = min(self._max_speed,
                                            self._velocity[j] + self._accel * dt)

                step = self._velocity[j] * dt * sign
                if abs(step) > dist:
                    step = diff

                self._position[j] += step

            if all_arrived:
                self._target = None
                self._velocity = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
                self._settled = True

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
            self._velocity = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}

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
