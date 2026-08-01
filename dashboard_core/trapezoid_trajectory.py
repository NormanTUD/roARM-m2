"""Trapezoidal velocity-profile trajectory for waypoint recordings.

Each waypoint is a single joint-space pose (b, s, e, h). Between two
consecutive waypoints, every joint runs an independent trapezoidal
velocity profile (acceleration -> cruise -> deceleration), with all
joints synchronized to arrive at the same time. The dominant joint
(the one that needs the longest minimum time) determines the segment
duration; smaller-distance joints are stretched to the same duration
so the whole arm arrives simultaneously with smooth, predictable motion.

This is the "most useful, most clear, direct path" between two
captured waypoints: linear interpolation in joint space with a
bounded-jerk velocity profile. The internal motion is optimized
automatically; only the endpoints are saved.
"""

import numpy as np


class TrapezoidTrajectory:
    """Joint-space trapezoidal velocity trajectory between discrete waypoints."""

    JOINTS = ("b", "s", "e", "h")

    def __init__(self, waypoints: list, v_max: float = 90.0,
                 a_max: float = 220.0, speed_factor: float = 1.0,
                 dwell_s: float = 0.0):
        if not waypoints or len(waypoints) < 1:
            raise ValueError("Need at least one waypoint")

        self._waypoints = waypoints
        self._v_max = max(0.1, v_max) * max(0.01, speed_factor)
        self._a_max = max(0.1, a_max)
        self._dwell_s = max(0.0, dwell_s)

        self._segments = []
        self._total_duration = 0.0
        self._segment_boundaries = [0.0]
        self._build_segments()

    # ------------------------------------------------------------------
    # Segment building
    # ------------------------------------------------------------------

    def _build_segments(self):
        starts = self._waypoints
        if len(starts) == 1:
            return

        for i in range(len(starts) - 1):
            a = starts[i]
            b = starts[i + 1]
            deltas = {j: b[j] - a[j] for j in self.JOINTS}
            distances = {j: abs(deltas[j]) for j in self.JOINTS}
            max_dist = max(distances.values())

            if max_dist < 1e-4:
                t_seg = 0.0
                profile = {j: _ZeroProfile() for j in self.JOINTS}
            else:
                t_min = max(
                    self._min_time_for(d) for d in distances.values()
                )
                t_seg = t_min

                profile = {}
                for j in self.JOINTS:
                    d = distances[j]
                    if d < 1e-4:
                        profile[j] = _ZeroProfile()
                        continue
                    profile[j] = _TrapProfile(
                        d=d,
                        t_total=t_seg,
                        a_max=self._a_max,
                        start=a[j],
                        sign=1.0 if deltas[j] >= 0 else -1.0,
                    )

            self._segments.append({
                "start": a,
                "end": b,
                "duration": t_seg,
                "profile": profile,
            })
            self._total_duration += t_seg
            self._total_duration += self._dwell_s
            self._segment_boundaries.append(self._total_duration)

    def _min_time_for(self, d: float) -> float:
        v = self._v_max
        a = self._a_max
        v_sq_over_a = (v * v) / a
        if d >= v_sq_over_a:
            return v / a + d / v
        return 2.0 * np.sqrt(d / a)

    # ------------------------------------------------------------------
    # Public interface (compatible with SmoothTrajectory)
    # ------------------------------------------------------------------

    def get_duration(self) -> float:
        return self._total_duration

    def get_waypoint_count(self) -> int:
        return len(self._waypoints)

    def get_segment_durations(self) -> list:
        return [s["duration"] for s in self._segments]

    def get_dwell_s(self) -> float:
        return self._dwell_s

    def sample(self, t: float) -> dict:
        t = float(np.clip(t, 0.0, self._total_duration))
        seg_idx = self._locate_segment(t)
        seg = self._segments[seg_idx]
        local_t = t - self._segment_boundaries[seg_idx]

        out = {}
        for j in self.JOINTS:
            out[j] = round(seg["profile"][j].position(local_t), 2)
        return out

    def get_speed_at(self, t: float) -> float:
        if self._total_duration <= 0:
            return 1.0
        t = float(np.clip(t, 0.0, self._total_duration))
        seg_idx = self._locate_segment(t)
        seg = self._segments[seg_idx]
        local_t = t - self._segment_boundaries[seg_idx]

        if local_t >= seg["duration"]:
            return 0.0

        peak = 0.0
        for j in self.JOINTS:
            v = abs(seg["profile"][j].velocity(local_t))
            if v > peak:
                peak = v
        if peak <= 0:
            return 0.0
        return min(1.2, max(0.7, peak / self._v_max))

    def get_segment_info(self, t: float) -> dict:
        seg_idx = self._locate_segment(t)
        return {
            "index": seg_idx,
            "total_segments": len(self._segments),
            "local_t": t - self._segment_boundaries[seg_idx],
            "duration": self._segments[seg_idx]["duration"],
            "start": self._segments[seg_idx]["start"],
            "end": self._segments[seg_idx]["end"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_segment(self, t: float) -> int:
        if not self._segments:
            return 0
        boundaries = self._segment_boundaries
        lo, hi = 0, len(boundaries) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if boundaries[mid] <= t:
                lo = mid
            else:
                hi = mid
        return lo


class _TrapProfile:
    """Single-joint trapezoidal velocity profile over a fixed duration.

    Solves for the peak velocity v_peak that lets the joint traverse
    distance d in exactly t_total given a peak acceleration a_max.
    Resulting motion is either:
      - trapezoidal (accel + cruise + decel) when t_total is long enough, or
      - triangular (accel + decel, no cruise) when t_total is short.
    """

    def __init__(self, d: float, t_total: float, a_max: float,
                 start: float, sign: float):
        self.d = d
        self.t_total = t_total
        self.a_max = a_max
        self.start = start
        self.sign = sign

        # v = (a*T - sqrt(a^2*T^2 - 4*a*d)) / 2  (smaller root)
        a2T2 = a_max * a_max * t_total * t_total
        disc = a2T2 - 4.0 * a_max * d
        if disc < 0.0:
            disc = 0.0
        v_peak = (a_max * t_total - np.sqrt(disc)) / 2.0
        v_peak = max(v_peak, 0.0)

        if v_peak < 1e-6:
            self.v_peak = 0.0
            self.t_accel = 0.0
            self.t_cruise = t_total
            self.t_decel = 0.0
        else:
            self.v_peak = v_peak
            self.t_accel = v_peak / a_max
            self.t_decel = v_peak / a_max
            cruise = t_total - self.t_accel - self.t_decel
            self.t_cruise = max(0.0, cruise)

        self.d_accel = 0.5 * a_max * self.t_accel * self.t_accel

    def position(self, t: float) -> float:
        if t <= 0.0:
            return self.start
        if t >= self.t_total:
            return self.start + self.sign * self.d

        if self.v_peak < 1e-6:
            return self.start + self.sign * self.d * (t / max(self.t_total, 1e-9))

        if t < self.t_accel:
            return self.start + self.sign * 0.5 * self.a_max * t * t

        if t < self.t_accel + self.t_cruise:
            tau = t - self.t_accel
            return self.start + self.sign * (self.d_accel + self.v_peak * tau)

        tau = self.t_total - t
        return self.start + self.sign * (self.d - 0.5 * self.a_max * tau * tau)

    def velocity(self, t: float) -> float:
        if t <= 0.0 or t >= self.t_total:
            return 0.0
        if self.v_peak < 1e-6:
            return self.sign * (self.d / max(self.t_total, 1e-9))

        if t < self.t_accel:
            return self.sign * self.a_max * t
        if t < self.t_accel + self.t_cruise:
            return self.sign * self.v_peak
        tau = self.t_total - t
        return self.sign * self.a_max * tau


class _ZeroProfile:
    """Constant-position profile used for joints that don't move in a segment."""

    def position(self, t: float) -> float:
        return 0.0

    def velocity(self, t: float) -> float:
        return 0.0
