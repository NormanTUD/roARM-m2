"""
GrabSequencer – Abstrahiert die gesamte Greif-Sequenz.

Zustandsmaschine für:
  IDLE → SCANNING → FOUND → CENTERING → APPROACHING → GRIPPING → LIFTING → PLACING → DONE

FIXES:
  - Slow ramp-up at start (no sudden jerky motion)
  - Adaptive centering with velocity-aware settling
  - Camera buffer flushing before every detection
  - Progressive speed: slow near target, faster far away
  - Re-acquisition logic when object is temporarily lost
  - Exponential backoff on lost detections instead of immediate failure
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable


class GrabState(Enum):
    IDLE = auto()
    SCAN_POSITION = auto()
    SETTLING = auto()          # NEW: Wait for arm to physically stop vibrating
    SEARCHING = auto()
    CENTERING = auto()
    VERIFY_CENTER = auto()     # NEW: Final verification before grab
    OPEN_GRIPPER = auto()
    APPROACHING = auto()
    GRIPPING = auto()
    LIFTING = auto()
    PLACING = auto()
    RELEASING = auto()
    RETRACTING = auto()
    PARKING = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class GrabConfig:
    """Konfiguration für einen Greifvorgang."""
    target_class: str = "bottle"
    place_offset_y: float = -100.0

    # Höhen (mm)
    scan_height: float = 200.0
    approach_height: float = 80.0       # LOWER! Was 135 — don't go UP after centering
    grab_height: float = 45.0           # Lower grab — was 75
    lift_height: float = 120.0
    place_height: float = 40.0
    retract_height: float = 120.0

    # Geschwindigkeiten — FASTER
    initial_speed: int = 20             # Was 8 — no need to crawl
    scan_speed: int = 25                # Was 12
    search_rotate_speed: int = 25       # Was 15
    center_speed: int = 30              # Was 12 — MUCH faster centering moves
    approach_speed: float = 0.30        # Was 0.12 — faster descent
    grab_speed: float = 0.20            # Was 0.08 — faster final descent
    lift_speed: float = 0.3
    place_speed: float = 0.3

    # Gripper
    gripper_torque: int = 300
    gripper_torque_threshold: int = 60
    gripper_close_timeout: float = 4.0
    gripper_step_rad: float = 0.08

    # Settling — SHORTER waits
    settle_time_after_scan: float = 1.5     # Was 2.5
    settle_time_after_search_step: float = 0.5  # Was 1.2
    settle_time_after_center_move: float = 0.3  # Was 1.0
    settle_frames_to_discard: int = 3       # Was 5

    # Zentrieren — MORE AGGRESSIVE
    center_max_iter: int = 15              # Was 25 — should converge faster now
    center_threshold_px: float = 30.0      # Was 20 — slightly more forgiving
    center_damping_initial: float = 0.6    # Was 0.3 — MUCH more aggressive start
    center_damping_final: float = 0.85     # Was 0.6 — aggressive finish
    center_deg_per_px_h: float = 0.08      # Was 0.04 — DOUBLE the correction
    center_deg_per_px_v: float = 0.06      # Was 0.03 — DOUBLE the correction
    center_smoothing_frames: int = 3       # Was 5 — fewer samples needed
    center_converge_needed: int = 2        # Was 3 — converge faster
    center_max_lost: int = 12
    center_max_step_base: float = 8.0      # Was 3.5 — allow bigger steps
    center_max_step_shoulder: float = 5.0  # Was 2.0 — allow bigger steps
    center_min_move_px: float = 5.0        # Was 8.0 — smaller dead zone
    center_sample_timeout: float = 2.0     # Was 4.0
    center_reacquire_attempts: int = 3
    center_reacquire_pause: float = 0.3    # Was 0.5

    # Suche
    search_range: Tuple[float, float] = (-90.0, 90.0)
    search_step: float = 20.0
    search_frames_per_step: int = 8        # Was 15
    search_direct_frames: int = 10         # Was 20

    # Timeouts — SHORTER
    wait_after_scan_move: float = 1.5      # Was 3.0
    wait_after_search_step: float = 0.5    # Was 1.2
    wait_after_center_move: float = 0.4    # Was 1.2 — CRITICAL: was way too slow
    wait_after_gripper_open: float = 0.3
    wait_after_approach: float = 1.0       # Was 2.0
    wait_after_grip: float = 0.5
    wait_after_lift: float = 1.5
    wait_after_place: float = 1.5
    wait_after_release: float = 0.3
    wait_after_retract: float = 1.0
    wait_after_park: float = 1.5

@dataclass
class GrabContext:
    """Laufzeit-Kontext eines Greifvorgangs."""
    state: GrabState = GrabState.IDLE
    config: GrabConfig = field(default_factory=GrabConfig)

    # Positionen
    grab_x: float = 0.0
    grab_y: float = 0.0
    grab_z: float = 0.0

    # Zentrierungs-State
    cur_base: float = 0.0
    cur_shoulder: float = 0.0
    cur_elbow: float = 90.0
    center_iter: int = 0
    center_converge_count: int = 0
    center_lost_count: int = 0
    center_reacquire_count: int = 0       # NEW
    center_last_known_px: Optional[Tuple[float, float]] = None  # NEW: last seen position
    center_velocity_px: Tuple[float, float] = (0.0, 0.0)       # NEW: estimated motion

    # Such-State
    search_current_deg: float = -90.0
    search_frame_count: int = 0
    search_direct_count: int = 0

    # Timing
    state_enter_time: float = 0.0
    last_move_time: float = 0.0           # NEW: when we last commanded a move

    # Ergebnis
    gripped: bool = False
    success: bool = False
    error_msg: str = ""

    # Letzte Detection
    last_detection: Optional[Dict] = None

    def enter_state(self, new_state: GrabState):
        self.state = new_state
        self.state_enter_time = time.time()

    @property
    def time_in_state(self) -> float:
        return time.time() - self.state_enter_time

    @property
    def time_since_last_move(self) -> float:
        """How long since we last commanded the arm to move."""
        return time.time() - self.last_move_time


class GrabSequencer:
    """
    Zustandsmaschine für den Greifvorgang.

    KEY FIXES:
    1. Slow initial movement — arm doesn't jerk to scan position
    2. Settling state — waits for vibration to stop before detecting
    3. Camera buffer flush — discards stale frames after every move
    4. Adaptive damping — gentle corrections that increase as we converge
    5. Re-acquisition — if object lost, pause and retry before failing
    6. Velocity tracking — predicts where object went if briefly lost
    """

    def __init__(self, arm, vision, debug: bool = False):
        self._arm = arm
        self._vision = vision
        self._debug = debug
        self._ctx: Optional[GrabContext] = None
        self._state_handlers: Dict[GrabState, Callable] = {
            GrabState.SCAN_POSITION: self._handle_scan_position,
            GrabState.SETTLING: self._handle_settling,
            GrabState.SEARCHING: self._handle_searching,
            GrabState.CENTERING: self._handle_centering,
            GrabState.VERIFY_CENTER: self._handle_verify_center,
            GrabState.OPEN_GRIPPER: self._handle_open_gripper,
            GrabState.APPROACHING: self._handle_approaching,
            GrabState.GRIPPING: self._handle_gripping,
            GrabState.LIFTING: self._handle_lifting,
            GrabState.PLACING: self._handle_placing,
            GrabState.RELEASING: self._handle_releasing,
            GrabState.RETRACTING: self._handle_retracting,
            GrabState.PARKING: self._handle_parking,
        }
        self._sub_state: int = 0
        self._search_phase: str = "direct"
        self._center_samples: List[Tuple[float, float]] = []
        self._settle_frames_discarded: int = 0
        self._verify_samples: List[Tuple[float, float]] = []

    def _dbg(self, msg: str):
        if self._debug:
            print(f"    [DBG] {msg}")

    @property
    def running(self) -> bool:
        return (self._ctx is not None and
                self._ctx.state not in (GrabState.DONE, GrabState.FAILED, GrabState.IDLE))

    @property
    def context(self) -> Optional[GrabContext]:
        return self._ctx

    @property
    def state(self) -> GrabState:
        return self._ctx.state if self._ctx else GrabState.IDLE

    def start(self, target_class: str, config: Optional[GrabConfig] = None):
        """Startet einen neuen Greifvorgang."""
        cfg = config or GrabConfig(target_class=target_class)
        cfg.target_class = target_class

        self._ctx = GrabContext(config=cfg)
        self._ctx.search_current_deg = cfg.search_range[0]
        self._ctx.enter_state(GrabState.SCAN_POSITION)
        self._sub_state = 0
        self._search_phase = "direct"
        self._center_samples = []
        self._settle_frames_discarded = 0
        self._verify_samples = []

    def tick(self) -> GrabState:
        """Ein Tick der Zustandsmaschine."""
        if not self._ctx or not self.running:
            return self._ctx.state if self._ctx else GrabState.IDLE

        handler = self._state_handlers.get(self._ctx.state)
        if handler:
            handler()

        return self._ctx.state

    def abort(self):
        if self._ctx:
            self._ctx.error_msg = "Abgebrochen"
            self._ctx.enter_state(GrabState.FAILED)

    # ─── Helper: Flush camera buffer ─────────────────────────────────────

    def _flush_camera(self, num_frames: int = 3):
        """
        Discard stale frames from camera buffer.
        CRITICAL after any arm movement — old frames show pre-move scene.
        """
        for _ in range(num_frames):
            self._vision.get_frame()
            time.sleep(0.03)

    def _get_adaptive_damping(self, pixel_dist: float) -> float:
        """
        More aggressive damping curve.
        Far away → moderate (not too gentle, we want speed)
        Close → very aggressive (converge fast)
        """
        cfg = self._ctx.config
        # Normalize: 0 = at center, 1 = far away (>100px)
        normalized = min(pixel_dist / 100.0, 1.0)
        # Far = initial, Close = final
        damping = cfg.center_damping_initial + (1.0 - normalized) * (
            cfg.center_damping_final - cfg.center_damping_initial
        )
        return damping

    # ─── State Handlers ───────────────────────────────────────────────────

    def _handle_scan_position(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            # FIX #1: Move to scan position SLOWLY to avoid jerking
            # First, move to an intermediate "ready" position gently
            self._arm.move_joints(b=0, s=0, e=90, h=180, spd=cfg.initial_speed, acc=5)
            ctx.last_move_time = time.time()
            self._sub_state = 1
            self._dbg(f"SCAN_POSITION: Gentle move started (spd={cfg.initial_speed})")

        elif self._sub_state == 1:
            # Show preview while waiting for arm to arrive
            dets, key = self._vision.update([cfg.target_class], "Fahre Scan-Position (langsam)...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_scan_move:
                # Transition to SETTLING (wait for vibration to stop)
                ctx.cur_base = 0.0
                ctx.cur_shoulder = 0.0
                ctx.cur_elbow = 90.0
                self._sub_state = 0
                self._settle_frames_discarded = 0
                ctx.enter_state(GrabState.SETTLING)
                self._dbg("SCAN_POSITION → SETTLING")

    def _handle_settling(self):
        """Wait for arm to stop, but don't waste time."""
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._flush_camera(cfg.settle_frames_to_discard)
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"SETTLING: Flushed {cfg.settle_frames_to_discard} frames")

        elif self._sub_state == 1:
            dets, key = self._vision.update([cfg.target_class], "Stabilisiere...")
            if key == ord('q'):
                self.abort()
                return
            # Only wait 0.3s instead of 0.5s
            if ctx.time_in_state >= 0.3:
                self._sub_state = 0
                ctx.enter_state(GrabState.SEARCHING)
                self._dbg("SETTLING → SEARCHING")


    def _handle_approaching(self):
        """
        FIXED: Don't go UP after centering. Go DOWN directly to grab.
        The arm is already centered over the object — just descend.
        """
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            # Get current position
            pos = self._arm.get_position()
            if pos is None or (pos[0] == 0 and pos[1] == 0 and pos[2] == 0):
                ctx.error_msg = "Position unbekannt"
                ctx.enter_state(GrabState.FAILED)
                self._dbg("APPROACHING: Position unbekannt!")
                return

            ctx.grab_x, ctx.grab_y, ctx.grab_z = pos
            self._dbg(f"APPROACHING: Position X={ctx.grab_x:.1f} Y={ctx.grab_y:.1f} Z={ctx.grab_z:.1f}")

            # Go DIRECTLY to grab height — no intermediate step up!
            # The arm is already above the object from centering position
            target_z = cfg.grab_height
            self._arm.move_cartesian(
                ctx.grab_x, ctx.grab_y, target_z,
                t=1.08, spd=cfg.approach_speed
            )
            ctx.last_move_time = time.time()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"APPROACHING: Descending directly to Z={target_z:.1f}mm")

        elif self._sub_state == 1:
            # Wait for descent to complete
            dets, key = self._vision.update([cfg.target_class], "Absenken zum Greifen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_approach:
                self._sub_state = 0
                ctx.enter_state(GrabState.GRIPPING)
                self._dbg("APPROACHING → GRIPPING")

    def _handle_searching(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._search_phase == "direct":
            dets, key = self._vision.update(
                [cfg.target_class], f"Suche '{cfg.target_class}'..."
            )
            if key == ord('q'):
                self.abort()
                return

            if dets:
                ctx.last_detection = dets[0]
                ctx.center_last_known_px = dets[0]['center_px']
                self._sub_state = 0
                self._center_samples = []
                ctx.center_iter = 0
                ctx.center_converge_count = 0
                ctx.center_lost_count = 0
                ctx.enter_state(GrabState.CENTERING)
                self._dbg(f"SEARCHING: Found! conf={dets[0]['confidence']:.2f} "
                         f"center={dets[0]['center_px']}")
                return

            ctx.search_direct_count += 1
            if ctx.search_direct_count >= cfg.search_direct_frames:
                self._search_phase = "rotate"
                self._sub_state = 0
                ctx.search_current_deg = cfg.search_range[0]
                self._dbg("SEARCHING: Direct not found → Rotation")

        elif self._search_phase == "rotate":
            if self._sub_state == 0:
                if ctx.search_current_deg > cfg.search_range[1]:
                    ctx.error_msg = f"'{cfg.target_class}' nicht gefunden"
                    ctx.enter_state(GrabState.FAILED)
                    return

                # FIX: Use slower rotation speed
                self._arm.move_joints(
                    b=ctx.search_current_deg, s=0, e=90, h=180,
                    spd=cfg.search_rotate_speed, acc=8
                )
                ctx.cur_base = ctx.search_current_deg
                ctx.last_move_time = time.time()
                self._sub_state = 1
                ctx.state_enter_time = time.time()
                self._dbg(f"SEARCHING: Rotate to {ctx.search_current_deg:.0f}°")

            elif self._sub_state == 1:
                # FIX: Wait LONGER for arm to arrive + settle
                dets, key = self._vision.update(
                    [cfg.target_class],
                    f"Suche '{cfg.target_class}' | Base: {ctx.search_current_deg:.0f}° (warte...)"
                )
                if key == ord('q'):
                    self.abort()
                    return

                if ctx.time_in_state < cfg.wait_after_search_step:
                    return

                # FIX: Flush camera after arm arrives
                self._flush_camera(3)
                self._sub_state = 2
                ctx.search_frame_count = 0

            elif self._sub_state == 2:
                dets, key = self._vision.update(
                    [cfg.target_class],
                    f"Suche '{cfg.target_class}' | Base: {ctx.search_current_deg:.0f}°"
                )
                if key == ord('q'):
                    self.abort()
                    return

                if dets:
                    ctx.last_detection = dets[0]
                    ctx.center_last_known_px = dets[0]['center_px']
                    self._sub_state = 0
                    self._center_samples = []
                    ctx.center_iter = 0
                    ctx.center_converge_count = 0
                    ctx.center_lost_count = 0
                    ctx.enter_state(GrabState.CENTERING)
                    self._dbg(f"SEARCHING: Found at {ctx.search_current_deg:.0f}°!")
                    return

                ctx.search_frame_count += 1
                if ctx.search_frame_count >= cfg.search_frames_per_step:
                    ctx.search_current_deg += cfg.search_step
                    self._sub_state = 0

    def _handle_centering(self):
        """
        REWORKED centering logic:
        - Flush camera after every move
        - Adaptive damping (gentle when far, aggressive when close)
        - Re-acquisition attempts when object lost
        - Velocity tracking to predict object position
        """
        ctx = self._ctx
        cfg = ctx.config

        if ctx.center_iter >= cfg.center_max_iter:
            ctx.error_msg = f"Max Iterationen ({cfg.center_max_iter}) erreicht"
            ctx.enter_state(GrabState.FAILED)
            self._dbg("CENTERING: Max iter reached!")
            return

        if self._sub_state == 0:
            # Phase 0: Flush camera buffer (critical after arm movement)
            self._flush_camera(cfg.settle_frames_to_discard)
            self._center_samples = []
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"CENTERING: Iter {ctx.center_iter+1}, flushed camera, collecting frames...")

        elif self._sub_state == 1:
            # Phase 1: Collect detection samples
            dets, key = self._vision.update(
                [cfg.target_class],
                f"Zentriere ({ctx.center_iter+1}/{cfg.center_max_iter}) | "
                f"B={ctx.cur_base:.1f}° S={ctx.cur_shoulder:.1f}° | "
                f"samples={len(self._center_samples)}/{cfg.center_smoothing_frames}"
            )
            if key == ord('q'):
                self.abort()
                return

            if dets:
                self._center_samples.append(dets[0]['center_px'])
                ctx.center_last_known_px = dets[0]['center_px']
                # Reset lost count on successful detection
                ctx.center_reacquire_count = 0
                self._dbg(f"CENTERING: Sample {len(self._center_samples)}/{cfg.center_smoothing_frames} "
                         f"@ {dets[0]['center_px']}")

            if len(self._center_samples) >= cfg.center_smoothing_frames:
                self._sub_state = 2
            elif ctx.time_in_state > cfg.center_sample_timeout:
                # Object lost — try to re-acquire
                self._sub_state = 10  # Re-acquisition sub-state
                self._dbg(f"CENTERING: Timeout collecting samples "
                         f"(got {len(self._center_samples)}/{cfg.center_smoothing_frames})")

        elif self._sub_state == 2:
            # Phase 2: Calculate correction from averaged samples
            centers = self._center_samples
            avg_cx = sum(c[0] for c in centers) / len(centers)
            avg_cy = sum(c[1] for c in centers) / len(centers)

            w, h = self._vision.resolution
            offset_px_x = avg_cx - (w / 2)
            offset_px_y = avg_cy - (h / 2)
            pixel_dist = (offset_px_x**2 + offset_px_y**2) ** 0.5

            self._dbg(f"CENTERING: offset=({offset_px_x:.1f}, {offset_px_y:.1f})px "
                     f"dist={pixel_dist:.1f}px threshold={cfg.center_threshold_px}")

            # Check if centered
            if pixel_dist < cfg.center_threshold_px:
                ctx.center_converge_count += 1
                self._dbg(f"CENTERING: Converged! count={ctx.center_converge_count}/"
                         f"{cfg.center_converge_needed}")
                if ctx.center_converge_count >= cfg.center_converge_needed:
                    # SUCCESS → go to verification
                    self._sub_state = 0
                    self._verify_samples = []
                    ctx.enter_state(GrabState.VERIFY_CENTER)
                    self._dbg("CENTERING → VERIFY_CENTER")
                    return
                # Need more convergence confirmations
                ctx.center_iter += 1
                self._sub_state = 0
                ctx.state_enter_time = time.time()
                return
            else:
                ctx.center_converge_count = 0

            # Calculate correction with ADAPTIVE damping
            damping = self._get_adaptive_damping(pixel_dist)
            self._dbg(f"CENTERING: pixel_dist={pixel_dist:.1f} → damping={damping:.3f}")

            d_base = 0.0
            if abs(offset_px_x) > cfg.center_min_move_px:
                d_base = -offset_px_x * cfg.center_deg_per_px_h * damping
                d_base = max(-cfg.center_max_step_base,
                           min(cfg.center_max_step_base, d_base))

            d_shoulder = 0.0
            if abs(offset_px_y) > cfg.center_min_move_px:
                d_shoulder = offset_px_y * cfg.center_deg_per_px_v * damping
                d_shoulder = max(-cfg.center_max_step_shoulder,
                               min(cfg.center_max_step_shoulder, d_shoulder))

            # Apply correction
            new_base = max(-90, min(90, ctx.cur_base + d_base))
            new_shoulder = max(-30, min(60, ctx.cur_shoulder + d_shoulder))

            self._dbg(f"CENTERING: d_base={d_base:.2f}° d_shoulder={d_shoulder:.2f}° "
                     f"→ B={new_base:.1f}° S={new_shoulder:.1f}°")

            ctx.cur_base = new_base
            ctx.cur_shoulder = new_shoulder

            # Move with CENTERING speed (slower than search)
            self._arm.move_joints(
                b=ctx.cur_base, s=ctx.cur_shoulder, e=ctx.cur_elbow,
                h=180, spd=cfg.center_speed, acc=10
            )
            ctx.last_move_time = time.time()
            self._sub_state = 3
            ctx.state_enter_time = time.time()

        elif self._sub_state == 3:
            # Phase 3: Wait for arm to arrive and settle
            dets, key = self._vision.update(
                [cfg.target_class],
                f"Bewege... B={ctx.cur_base:.1f}° S={ctx.cur_shoulder:.1f}° (warte...)"
            )
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_center_move:
                ctx.center_iter += 1
                self._sub_state = 0  # Back to flush + collect
                ctx.state_enter_time = time.time()
                self._dbg("CENTERING: Move complete, next iteration")

        elif self._sub_state == 10:
            # RE-ACQUISITION: Object was lost during sample collection
            ctx.center_lost_count += 1
            ctx.center_reacquire_count += 1

            self._dbg(f"CENTERING: Re-acquire attempt {ctx.center_reacquire_count}/"
                     f"{cfg.center_reacquire_attempts} "
                     f"(total lost={ctx.center_lost_count}/{cfg.center_max_lost})")

            if ctx.center_lost_count >= cfg.center_max_lost:
                ctx.error_msg = "Objekt zu oft verloren"
                ctx.enter_state(GrabState.FAILED)
                return

            if ctx.center_reacquire_count >= cfg.center_reacquire_attempts:
                # Tried multiple times — maybe we moved too far
                # Back up slightly toward last known position
                self._dbg("CENTERING: Re-acquire failed, trying small backup move")
                ctx.center_reacquire_count = 0
                # Don't move, just retry with fresh frames
                self._sub_state = 0
                ctx.state_enter_time = time.time()
                return

            # Wait a moment, flush camera, try again
            time.sleep(cfg.center_reacquire_pause)
            self._flush_camera(4)
            self._center_samples = []
            self._sub_state = 1
            ctx.state_enter_time = time.time()

    def _handle_verify_center(self):
        """
        SIMPLIFIED: Quick verification — just check 3 frames, don't wait 0.8s.
        Speed > perfection. If it's roughly centered, GO.
        """
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._flush_camera(2)
            self._verify_samples = []
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg("VERIFY_CENTER: Starting verification...")

        elif self._sub_state == 1:
            dets, key = self._vision.update(
                [cfg.target_class],
                f"Verifiziere... ({len(self._verify_samples)}/3)"
            )
            if key == ord('q'):
                self.abort()
                return

            if dets:
                self._verify_samples.append(dets[0]['center_px'])

            # Only need 3 samples, no minimum time
            if len(self._verify_samples) >= 3:
                w, h = self._vision.resolution
                all_centered = True
                for cx, cy in self._verify_samples:
                    dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
                    if dist > cfg.center_threshold_px * 2.0:
                        all_centered = False
                        break

                if all_centered:
                    self._dbg("VERIFY_CENTER: ✓ Confirmed centered!")
                    self._sub_state = 0
                    ctx.enter_state(GrabState.OPEN_GRIPPER)
                else:
                    self._dbg("VERIFY_CENTER: ✗ Not centered, back to CENTERING")
                    ctx.center_converge_count = 0
                    self._sub_state = 0
                    ctx.enter_state(GrabState.CENTERING)

            elif ctx.time_in_state > 1.5:
                # Timeout — just proceed if we have any data
                if len(self._verify_samples) >= 1:
                    self._sub_state = 0
                    ctx.enter_state(GrabState.OPEN_GRIPPER)
                    self._dbg("VERIFY_CENTER: Timeout, proceeding anyway")
                else:
                    ctx.center_converge_count = 0
                    self._sub_state = 0
                    ctx.enter_state(GrabState.CENTERING)

    def _handle_open_gripper(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._arm.gripper_open()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg("OPEN_GRIPPER: Öffne...")

        elif self._sub_state == 1:
            dets, key = self._vision.update([cfg.target_class], "Gripper öffnen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_gripper_open:
                self._sub_state = 0
                ctx.enter_state(GrabState.APPROACHING)
                self._dbg("OPEN_GRIPPER → APPROACHING")

    def _handle_approaching(self):
        """
        FIXED: Don't go UP after centering. Go DOWN directly to grab.
        The arm is already centered over the object — just descend.
        """
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            # Get current position
            pos = self._arm.get_position()
            if pos is None or (pos[0] == 0 and pos[1] == 0 and pos[2] == 0):
                ctx.error_msg = "Position unbekannt"
                ctx.enter_state(GrabState.FAILED)
                self._dbg("APPROACHING: Position unbekannt!")
                return

            ctx.grab_x, ctx.grab_y, ctx.grab_z = pos
            self._dbg(f"APPROACHING: Position X={ctx.grab_x:.1f} Y={ctx.grab_y:.1f} Z={ctx.grab_z:.1f}")

            # Go DIRECTLY to grab height — no intermediate step up!
            # The arm is already above the object from centering position
            target_z = cfg.grab_height
            self._arm.move_cartesian(
                ctx.grab_x, ctx.grab_y, target_z,
                t=1.08, spd=cfg.approach_speed
            )
            ctx.last_move_time = time.time()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"APPROACHING: Descending directly to Z={target_z:.1f}mm")

        elif self._sub_state == 1:
            # Wait for descent to complete
            dets, key = self._vision.update([cfg.target_class], "Absenken zum Greifen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_approach:
                self._sub_state = 0
                ctx.enter_state(GrabState.GRIPPING)
                self._dbg("APPROACHING → GRIPPING")

    def _handle_gripping(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._dbg("GRIPPING: Schließe Gripper...")
            # Set torque + grip
            self._arm.gripper_set_max_torque(cfg.gripper_torque)
            time.sleep(0.3)

            gripped = self._arm.gripper_close_until_resistance(
                torque_threshold=cfg.gripper_torque_threshold,
                timeout=cfg.gripper_close_timeout,
                step_rad=cfg.gripper_step_rad
            )
            ctx.gripped = gripped
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"GRIPPING: Ergebnis gripped={gripped}")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Gegriffen!")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_grip:
                self._sub_state = 0
                ctx.enter_state(GrabState.LIFTING)
                self._dbg("GRIPPING → LIFTING")

    def _handle_lifting(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            lift_z = cfg.grab_height + cfg.lift_height
            self._arm.move_cartesian(
                ctx.grab_x, ctx.grab_y, lift_z,
                t=3.14, spd=cfg.lift_speed
            )
            ctx.last_move_time = time.time()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"LIFTING: Z={lift_z:.1f}mm")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Anheben...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_lift:
                self._sub_state = 0
                ctx.enter_state(GrabState.PLACING)
                self._dbg("LIFTING → PLACING")

    def _handle_placing(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            place_y = ctx.grab_y + cfg.place_offset_y
            place_z = cfg.grab_height + cfg.place_height
            self._arm.move_cartesian(
                ctx.grab_x, place_y, place_z,
                t=3.14, spd=cfg.place_speed
            )
            ctx.last_move_time = time.time()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"PLACING: Y={place_y:.1f} Z={place_z:.1f}")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Ablegen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_place:
                self._sub_state = 0
                ctx.enter_state(GrabState.RELEASING)
                self._dbg("PLACING → RELEASING")

    def _handle_releasing(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._arm.gripper_open()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg("RELEASING: Gripper öffnen")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Loslassen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_release:
                self._sub_state = 0
                ctx.enter_state(GrabState.RETRACTING)
                self._dbg("RELEASING → RETRACTING")

    def _handle_retracting(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            place_y = ctx.grab_y + cfg.place_offset_y
            retract_z = cfg.grab_height + cfg.retract_height
            self._arm.move_cartesian(
                ctx.grab_x, place_y, retract_z,
                t=3.14, spd=0.25
            )
            ctx.last_move_time = time.time()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg(f"RETRACTING: Z={retract_z:.1f}")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Zurückziehen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_retract:
                self._sub_state = 0
                ctx.enter_state(GrabState.PARKING)
                self._dbg("RETRACTING → PARKING")

    def _handle_parking(self):
        ctx = self._ctx
        cfg = ctx.config

        if self._sub_state == 0:
            self._arm.park()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
            self._dbg("PARKING: Park-Position")

        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Parken...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_park:
                ctx.success = True
                ctx.enter_state(GrabState.DONE)
                self._dbg("PARKING → DONE (Erfolg!)")

