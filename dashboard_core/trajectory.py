import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

from .kinematics import (
    MIN_SPEED_FACTOR, MAX_SPEED_FACTOR,
    END_RAMP_PERCENT, START_RAMP_PERCENT,
)


class SmoothTrajectory:
    """Smooth time-continuous trajectory from discrete waypoints."""

    def __init__(self, waypoints: list, speed_factor: float = 1.0):
        self._waypoints = waypoints
        self._speed_factor = speed_factor
        self._splines = {}
        self._time_map = None
        self._t_new = None
        self._speed_profile = None
        self._total_duration = 0.0
        self._original_duration = 0.0
        self._build_splines()
        self._compute_adaptive_timing()

    def _build_splines(self):
        times = np.array([wp["t"] for wp in self._waypoints])
        if times[0] > 0.01:
            times = np.concatenate([[0.0, times[0] * 0.5], times])
        for joint in ["b", "s", "e", "h"]:
            values = np.array([wp[joint] for wp in self._waypoints])
            if len(times) > len(values):
                pad = np.array([values[0], values[0]])
                values = np.concatenate([pad, values])
            self._splines[joint] = CubicSpline(times, values, bc_type='clamped')
        self._original_duration = times[-1]

    def _compute_curvature(self, t_original: np.ndarray) -> np.ndarray:
        curvature = np.zeros(len(t_original))
        for joint in ["b", "s", "e", "h"]:
            d2 = self._splines[joint](t_original, 2)
            curvature += d2 ** 2
        curvature = np.sqrt(curvature)
        # FIX #2: Use a much wider smoothing kernel to eliminate
        # oscillations from densely-packed waypoints.
        # MR Ch.9: trajectory smoothness requires continuous acceleration.
        # A narrow kernel preserves spline oscillations; a wide one
        # produces a smooth speed profile that won't cause judder.
        kernel_size = max(40, len(t_original) // 10)
        return uniform_filter1d(curvature, size=kernel_size, mode='nearest')

    def _curvature_to_speed_profile(self, curvature: np.ndarray) -> np.ndarray:
        max_curv = np.percentile(curvature, 95) if curvature.max() > 0 else 1.0
        norm = np.clip(curvature / max(max_curv, 1e-6), 0, 1)
        # FIX #2b: Don't slow down as aggressively. The servo firmware
        # needs continuous motion to stay smooth. Going below 0.75 causes
        # the command rate to drop too low for smooth servo interpolation.
        effective_min = max(MIN_SPEED_FACTOR, 0.75)
        return MAX_SPEED_FACTOR - norm * (MAX_SPEED_FACTOR - effective_min)

    def _apply_ramps(self, speed_profile: np.ndarray) -> np.ndarray:
        n = len(speed_profile)
        end_start = int(n * (1.0 - END_RAMP_PERCENT))
        for i in range(end_start, n):
            progress = (i - end_start) / (n - end_start)
            speed_profile[i] = min(speed_profile[i],
                MIN_SPEED_FACTOR + (1.0 - progress) * (speed_profile[i] - MIN_SPEED_FACTOR))
        start_end = int(n * START_RAMP_PERCENT)
        for i in range(start_end):
            progress = i / max(start_end, 1)
            speed_profile[i] = MIN_SPEED_FACTOR + progress * (speed_profile[i] - MIN_SPEED_FACTOR)
        return speed_profile

    def _compute_adaptive_timing(self):
        n_samples = 500
        t_original = np.linspace(0, self._original_duration, n_samples)
        curvature = self._compute_curvature(t_original)
        speed_profile = self._curvature_to_speed_profile(curvature)
        speed_profile = self._apply_ramps(speed_profile)
        
        # FIX #2c: Smooth the speed profile itself to prevent rapid
        # speed changes that cause variable command rates.
        # MR Ch.9 Sec 9.2.2: time scaling must have continuous first derivative.
        speed_profile = uniform_filter1d(speed_profile, size=30, mode='nearest')
        speed_profile = np.clip(speed_profile, MIN_SPEED_FACTOR, MAX_SPEED_FACTOR)
        
        dt = t_original[1] - t_original[0]
        dt_new = dt / (speed_profile * self._speed_factor)
        t_new = np.cumsum(dt_new)
        t_new = np.insert(t_new, 0, 0.0)[:-1]
        self._total_duration = t_new[-1]
        self._t_new = t_new
        self._speed_profile = speed_profile
        self._time_map = CubicSpline(t_new, t_original, bc_type='natural')

    def get_duration(self) -> float:
        return self._total_duration

    def sample(self, t_playback: float) -> dict:
        t_playback = np.clip(t_playback, 0, self._total_duration)
        t_orig = float(self._time_map(t_playback))
        t_orig = np.clip(t_orig, 0, self._original_duration)
        return {j: round(float(self._splines[j](t_orig)), 2)
                for j in ["b", "s", "e", "h"]}

    def get_speed_at(self, t_playback: float) -> float:
        idx = np.searchsorted(self._t_new, t_playback)
        idx = min(idx, len(self._speed_profile) - 1)
        return self._speed_profile[idx]
