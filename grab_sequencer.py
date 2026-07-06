"""
GrabSequencer – Abstrahiert die gesamte Greif-Sequenz.

Zustandsmaschine für:
  IDLE → SCANNING → FOUND → CENTERING → APPROACHING → GRIPPING → LIFTING → PLACING → DONE

Jeder Zustand hat klare Ein-/Ausgangsbedingungen.
Der EyeInHandController muss nur noch update() callen.
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable


class GrabState(Enum):
    IDLE = auto()
    SCAN_POSITION = auto()      # Fahre in Scan-Stellung
    SEARCHING = auto()          # Suche Objekt (direkt + Rotation)
    CENTERING = auto()          # Zentriere Objekt in Bildmitte
    OPEN_GRIPPER = auto()       # Gripper öffnen
    APPROACHING = auto()        # Absenken zum Objekt
    GRIPPING = auto()           # Gripper schließen
    LIFTING = auto()            # Anheben
    PLACING = auto()            # Ablegen
    RELEASING = auto()          # Gripper öffnen zum Ablegen
    RETRACTING = auto()         # Zurückziehen
    PARKING = auto()            # Parkposition
    DONE = auto()               # Fertig (Erfolg)
    FAILED = auto()             # Fehlgeschlagen


@dataclass
class GrabConfig:
    """Konfiguration für einen Greifvorgang."""
    target_class: str = "bottle"
    place_offset_y: float = -100.0
    
    # Höhen (mm)
    scan_height: float = 200.0
    approach_height: float = 135.0      # Zwischenhöhe beim Absenken
    grab_height: float = 75.0           # Greifhöhe
    lift_height: float = 120.0          # Anheben nach Griff (relativ zu grab_height)
    place_height: float = 40.0          # Ablege-Höhe (relativ zu grab_height)
    retract_height: float = 120.0       # Rückzug-Höhe (relativ zu grab_height)
    
    # Geschwindigkeiten
    scan_speed: int = 20
    approach_speed: float = 0.15
    grab_speed: float = 0.1
    lift_speed: float = 0.2
    place_speed: float = 0.2
    
    # Gripper
    gripper_torque: int = 300
    gripper_torque_threshold: int = 60
    gripper_close_timeout: float = 4.0
    gripper_step_rad: float = 0.08
    
    # Zentrieren
    center_max_iter: int = 15
    center_threshold_px: float = 20.0
    center_damping: float = 0.6
    center_deg_per_px_h: float = 0.05
    center_deg_per_px_v: float = 0.035
    center_smoothing_frames: int = 3
    center_converge_needed: int = 2
    center_max_lost: int = 4
    center_max_step_base: float = 5.0
    center_max_step_shoulder: float = 3.0
    center_min_move_px: float = 5.0
    
    # Suche
    search_range: Tuple[float, float] = (-90.0, 90.0)
    search_step: float = 20.0
    search_frames_per_step: int = 10
    search_direct_frames: int = 15
    
    # Timeouts (Sekunden die in einem State gewartet wird)
    wait_after_scan_move: float = 2.0
    wait_after_search_step: float = 0.3
    wait_after_center_move: float = 0.6
    wait_after_gripper_open: float = 0.5
    wait_after_approach: float = 1.5
    wait_after_grip: float = 0.5
    wait_after_lift: float = 2.0
    wait_after_place: float = 2.0
    wait_after_release: float = 0.5
    wait_after_retract: float = 1.5
    wait_after_park: float = 2.0


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
    
    # Such-State
    search_current_deg: float = -90.0
    search_frame_count: int = 0
    search_direct_count: int = 0
    
    # Timing
    state_enter_time: float = 0.0
    
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


class GrabSequencer:
    """
    Zustandsmaschine für den Greifvorgang.
    
    Verwendung:
        seq = GrabSequencer(arm_interface, vision_interface)
        seq.start("bottle")
        while seq.running:
            seq.tick()  # Muss im Main-Thread laufen (wegen GUI)
    """
    
    def __init__(self, arm, vision):
        """
        arm: Objekt mit Methoden:
            - move_joints(b, s, e, h, spd, acc)
            - move_cartesian(x, y, z, t, spd)
            - gripper_open()
            - gripper_close_until_resistance(torque_threshold, timeout, step_rad)
            - gripper_set_max_torque(torque)
            - get_position() -> (x, y, z) oder None
            - park()
            
        vision: Objekt mit Methoden:
            - detect(target_classes) -> List[Dict]
            - update(target_classes, status_text) -> (detections, key)
            - update_for(seconds, target_classes, status_text)
            - resolution -> (w, h)
            - get_frame()
        """
        self._arm = arm
        self._vision = vision
        self._ctx: Optional[GrabContext] = None
        self._state_handlers: Dict[GrabState, Callable] = {
            GrabState.SCAN_POSITION: self._handle_scan_position,
            GrabState.SEARCHING: self._handle_searching,
            GrabState.CENTERING: self._handle_centering,
            GrabState.OPEN_GRIPPER: self._handle_open_gripper,
            GrabState.APPROACHING: self._handle_approaching,
            GrabState.GRIPPING: self._handle_gripping,
            GrabState.LIFTING: self._handle_lifting,
            GrabState.PLACING: self._handle_placing,
            GrabState.RELEASING: self._handle_releasing,
            GrabState.RETRACTING: self._handle_retracting,
            GrabState.PARKING: self._handle_parking,
        }
        # Sub-State für mehrstufige States
        self._sub_state: int = 0
        self._search_phase: str = "direct"  # "direct" oder "rotate"
    
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
    
    def tick(self) -> GrabState:
        """
        Ein Tick der Zustandsmaschine. 
        MUSS im Main-Thread laufen (wegen cv2.imshow).
        Gibt aktuellen State zurück.
        """
        if not self._ctx or not self.running:
            return self._ctx.state if self._ctx else GrabState.IDLE
        
        handler = self._state_handlers.get(self._ctx.state)
        if handler:
            handler()
        
        return self._ctx.state
    
    def abort(self):
        """Bricht den Vorgang ab."""
        if self._ctx:
            self._ctx.error_msg = "Abgebrochen"
            self._ctx.enter_state(GrabState.FAILED)
    
    # ─── State Handlers ───────────────────────────────────────────────────
    
    def _handle_scan_position(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            # Bewegung starten
            self._arm.move_joints(b=0, s=0, e=90, h=180, spd=cfg.scan_speed, acc=10)
            self._sub_state = 1
        
        elif self._sub_state == 1:
            # Warten + Live-Preview
            dets, key = self._vision.update([cfg.target_class], "Fahre Scan-Position...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_scan_move:
                ctx.cur_base = 0.0
                ctx.cur_shoulder = 0.0
                ctx.cur_elbow = 90.0
                self._sub_state = 0
                ctx.enter_state(GrabState.SEARCHING)
    
    def _handle_searching(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._search_phase == "direct":
            # Direkt schauen
            dets, key = self._vision.update(
                [cfg.target_class], f"Suche '{cfg.target_class}'..."
            )
            if key == ord('q'):
                self.abort()
                return
            
            if dets:
                ctx.last_detection = dets[0]
                self._sub_state = 0
                ctx.enter_state(GrabState.CENTERING)
                return
            
            ctx.search_direct_count += 1
            if ctx.search_direct_count >= cfg.search_direct_frames:
                # Wechsel zu Rotation
                self._search_phase = "rotate"
                self._sub_state = 0
                ctx.search_current_deg = cfg.search_range[0]
        
        elif self._search_phase == "rotate":
            if self._sub_state == 0:
                # Nächste Position anfahren
                if ctx.search_current_deg > cfg.search_range[1]:
                    # Gesamter Bereich abgesucht
                    ctx.error_msg = f"'{cfg.target_class}' nicht gefunden"
                    ctx.enter_state(GrabState.FAILED)
                    return
                
                self._arm.move_joints(
                    b=ctx.search_current_deg, s=0, e=90, h=180, spd=25, acc=10
                )
                ctx.cur_base = ctx.search_current_deg
                self._sub_state = 1
                ctx.state_enter_time = time.time()  # Reset timer
            
            elif self._sub_state == 1:
                # Warten bis Arm da ist
                dets, key = self._vision.update(
                    [cfg.target_class],
                    f"Suche '{cfg.target_class}' | Base: {ctx.search_current_deg:.0f}°"
                )
                if key == ord('q'):
                    self.abort()
                    return
                
                if ctx.time_in_state < cfg.wait_after_search_step:
                    return  # Noch warten
                
                self._sub_state = 2
                ctx.search_frame_count = 0
            
            elif self._sub_state == 2:
                # Frames checken
                dets, key = self._vision.update(
                    [cfg.target_class],
                    f"Suche '{cfg.target_class}' | Base: {ctx.search_current_deg:.0f}°"
                )
                if key == ord('q'):
                    self.abort()
                    return
                
                if dets:
                    ctx.last_detection = dets[0]
                    self._sub_state = 0
                    ctx.enter_state(GrabState.CENTERING)
                    return
                
                ctx.search_frame_count += 1
                if ctx.search_frame_count >= cfg.search_frames_per_step:
                    # Nächster Schritt
                    ctx.search_current_deg += cfg.search_step
                    self._sub_state = 0
    
    def _handle_centering(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if ctx.center_iter >= cfg.center_max_iter:
            ctx.error_msg = f"Max Iterationen ({cfg.center_max_iter}) erreicht"
            ctx.enter_state(GrabState.FAILED)
            return
        
        if self._sub_state == 0:
            # Detection sammeln
            self._center_samples = []
            self._sub_state = 1
        
        elif self._sub_state == 1:
            # Frames sammeln
            dets, key = self._vision.update(
                [cfg.target_class],
                f"Zentriere (Iter {ctx.center_iter+1}/{cfg.center_max_iter}) | "
                f"B={ctx.cur_base:.1f}°"
            )
            if key == ord('q'):
                self.abort()
                return
            
            if dets:
                self._center_samples.append(dets[0]['center_px'])
            
            if len(self._center_samples) >= cfg.center_smoothing_frames:
                self._sub_state = 2
            elif ctx.time_in_state > 2.0:
                # Timeout beim Sammeln
                ctx.center_lost_count += 1
                if ctx.center_lost_count >= cfg.center_max_lost:
                    ctx.error_msg = "Objekt zu oft verloren"
                    ctx.enter_state(GrabState.FAILED)
                    return
                ctx.center_iter += 1
                self._sub_state = 0
                ctx.enter_state(GrabState.CENTERING)  # Reset timer
        
        elif self._sub_state == 2:
            # Offset berechnen
            centers = self._center_samples
            avg_cx = sum(c[0] for c in centers) / len(centers)
            avg_cy = sum(c[1] for c in centers) / len(centers)
            
            w, h = self._vision.resolution
            offset_px_x = avg_cx - (w / 2)
            offset_px_y = avg_cy - (h / 2)
            pixel_dist = (offset_px_x**2 + offset_px_y**2) ** 0.5
            
            # Zentriert?
            if pixel_dist < cfg.center_threshold_px:
                ctx.center_converge_count += 1
                if ctx.center_converge_count >= cfg.center_converge_needed:
                    # ERFOLG → Gripper öffnen
                    self._sub_state = 0
                    ctx.enter_state(GrabState.OPEN_GRIPPER)
                    return
                # Nochmal verifizieren
                ctx.center_iter += 1
                self._sub_state = 0
                ctx.enter_state(GrabState.CENTERING)
                return
            else:
                ctx.center_converge_count = 0
            
            # Korrektur berechnen
            d_base = 0.0
            if abs(offset_px_x) > cfg.center_min_move_px:
                d_base = -offset_px_x * cfg.center_deg_per_px_h * cfg.center_damping
                d_base = max(-cfg.center_max_step_base, 
                           min(cfg.center_max_step_base, d_base))
            
            d_shoulder = 0.0
            if abs(offset_px_y) > cfg.center_min_move_px:
                d_shoulder = offset_px_y * cfg.center_deg_per_px_v * cfg.center_damping
                d_shoulder = max(-cfg.center_max_step_shoulder,
                               min(cfg.center_max_step_shoulder, d_shoulder))
            
            # Neue Winkel
            ctx.cur_base = max(-90, min(90, ctx.cur_base + d_base))
            ctx.cur_shoulder = max(-30, min(60, ctx.cur_shoulder + d_shoulder))
            
            # Bewegen
            self._arm.move_joints(
                b=ctx.cur_base, s=ctx.cur_shoulder, e=ctx.cur_elbow, 
                h=180, spd=20, acc=10
            )
            self._sub_state = 3
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 3:
            # Warten nach Bewegung
            dets, key = self._vision.update(
                [cfg.target_class], "Bewege..."
            )
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_center_move:
                ctx.center_iter += 1
                ctx.center_lost_count = 0
                self._sub_state = 0
                ctx.enter_state(GrabState.CENTERING)
    
    def _handle_open_gripper(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            self._arm.gripper_open()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update([cfg.target_class], "Gripper öffnen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_gripper_open:
                self._sub_state = 0
                ctx.enter_state(GrabState.APPROACHING)
    
    def _handle_approaching(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            # Position holen
            pos = self._arm.get_position()
            if pos is None or (pos[0] == 0 and pos[1] == 0 and pos[2] == 0):
                ctx.error_msg = "Position unbekannt"
                ctx.enter_state(GrabState.FAILED)
                return
            
            ctx.grab_x, ctx.grab_y, ctx.grab_z = pos
            
            # Zwischenhöhe anfahren
            self._arm.move_cartesian(
                ctx.grab_x, ctx.grab_y, cfg.approach_height, 
                t=1.08, spd=cfg.approach_speed
            )
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            # Warten (Zwischenhöhe)
            dets, key = self._vision.update([cfg.target_class], "Absenken (Zwischen)...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_approach:
                # Greifhöhe anfahren
                self._arm.move_cartesian(
                    ctx.grab_x, ctx.grab_y, cfg.grab_height,
                    t=1.08, spd=cfg.grab_speed
                )
                self._sub_state = 2
                ctx.state_enter_time = time.time()
        
        elif self._sub_state == 2:
            # Warten (Greifhöhe)
            dets, key = self._vision.update([cfg.target_class], "Greifposition...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_approach:
                self._sub_state = 0
                ctx.enter_state(GrabState.GRIPPING)
    
    def _handle_gripping(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            # Torque setzen + Greifen
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
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Gegriffen!")
            if ctx.time_in_state >= cfg.wait_after_grip:
                self._sub_state = 0
                ctx.enter_state(GrabState.LIFTING)
    
    def _handle_lifting(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            self._arm.move_cartesian(
                ctx.grab_x, ctx.grab_y, cfg.grab_height + cfg.lift_height,
                t=3.14, spd=cfg.lift_speed
            )
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Anheben...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_lift:
                self._sub_state = 0
                ctx.enter_state(GrabState.PLACING)
    
    def _handle_placing(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            place_y = ctx.grab_y + cfg.place_offset_y
            self._arm.move_cartesian(
                ctx.grab_x, place_y, cfg.grab_height + cfg.place_height,
                t=3.14, spd=cfg.place_speed
            )
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Ablegen...")
            if key == ord('q'):
                self.abort()
                return
            if ctx.time_in_state >= cfg.wait_after_place:
                self._sub_state = 0
                ctx.enter_state(GrabState.RELEASING)
    
    def _handle_releasing(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            self._arm.gripper_open()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Loslassen...")
            if ctx.time_in_state >= cfg.wait_after_release:
                self._sub_state = 0
                ctx.enter_state(GrabState.RETRACTING)
    
    def _handle_retracting(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            place_y = ctx.grab_y + cfg.place_offset_y
            self._arm.move_cartesian(
                ctx.grab_x, place_y, cfg.grab_height + cfg.retract_height,
                t=3.14, spd=0.25
            )
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Zurückziehen...")
            if ctx.time_in_state >= cfg.wait_after_retract:
                self._sub_state = 0
                ctx.enter_state(GrabState.PARKING)
    
    def _handle_parking(self):
        ctx = self._ctx
        cfg = ctx.config
        
        if self._sub_state == 0:
            self._arm.park()
            self._sub_state = 1
            ctx.state_enter_time = time.time()
        
        elif self._sub_state == 1:
            dets, key = self._vision.update(status_text="Parken...")
            if ctx.time_in_state >= cfg.wait_after_park:
                ctx.success = True
                ctx.enter_state(GrabState.DONE)
