#!/usr/bin/env python3
"""safety.py - RoArm-M2-S Hardware-Schutzschicht

Verhindert:
- Befehle an physisch unmögliche Positionen
- Unkontrolliertes Streaming ohne Feedback
- Überhitzung durch Dauerlast
- Müll-Reads die zu falschen Korrekturen führen
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# SICHERHEITS-GRENZEN (anpassen an deinen Arm!)
# ============================================================

@dataclass
class SafetyLimits:
    """Physische Grenzen des RoArm-M2-S."""
    
    # Absolute Gelenkgrenzen (Grad) - NIEMALS überschreiten
    b_min: float = -135.0
    b_max: float = 135.0
    s_min: float = -30.0
    s_max: float = 90.0
    e_min: float = 0.0
    e_max: float = 180.0
    h_min: float = 0.0
    h_max: float = 360.0
    
    # Maximale Geschwindigkeit/Beschleunigung
    max_spd: int = 50
    max_acc: int = 30
    
    # Maximale Änderung pro Befehl (Grad) - verhindert Sprünge
    max_delta_per_cmd: float = 15.0  # Max 15° pro einzelnem Befehl
    
    # Streaming-Limits
    max_stream_hz: int = 50
    max_stream_duration_s: float = 120.0  # Max 2 Minuten Streaming
    
    # Überhitzungsschutz
    max_continuous_move_s: float = 60.0  # Max 60s ohne Pause
    cooldown_pause_s: float = 5.0        # Pflichtpause danach
    
    # Plausibilitäts-Grenzen für gelesene Werte
    max_plausible_position: float = 200.0  # Kein Gelenk > 200°
    max_plausible_error: float = 10.0      # Fehler > 10° = Müll-Read


# ============================================================
# POSITIONS-VALIDATOR
# ============================================================

class PositionValidator:
    """Prüft ob Positionen und Befehle sicher sind."""
    
    def __init__(self, limits: SafetyLimits = None):
        self.limits = limits or SafetyLimits()
        self._last_commanded = None
        self._stream_start_time = None
        self._continuous_move_start = None
        self._total_commands_sent = 0
        self._emergency_stop = False
    
    def validate_target(self, b: float, s: float, e: float, h: float) -> tuple[bool, str]:
        """
        Prüft ob eine Zielposition sicher ist.
        Returns: (is_safe, reason)
        """
        if self._emergency_stop:
            return False, "EMERGENCY STOP AKTIV"
        
        L = self.limits
        
        # 1. Absolute Grenzen
        if not (L.b_min <= b <= L.b_max):
            return False, f"b={b:.2f}° außerhalb [{L.b_min}, {L.b_max}]"
        if not (L.s_min <= s <= L.s_max):
            return False, f"s={s:.2f}° außerhalb [{L.s_min}, {L.s_max}]"
        if not (L.e_min <= e <= L.e_max):
            return False, f"e={e:.2f}° außerhalb [{L.e_min}, {L.e_max}]"
        if not (L.h_min <= h <= L.h_max):
            return False, f"h={h:.2f}° außerhalb [{L.h_min}, {L.h_max}]"
        
        # 2. Sprung-Erkennung - NUR gegen letzten GESENDETEN Befehl
        #    UND nur wenn der letzte Befehl kürzlich war (< 1s)
        if self._last_commanded is not None:
            time_since_last = time.time() - self._last_commanded["t"]
            
            delta_b = abs(b - self._last_commanded["b"])
            delta_s = abs(s - self._last_commanded["s"])
            delta_e = abs(e - self._last_commanded["e"])
            delta_h = abs(h - self._last_commanded["h"])
            max_delta = max(delta_b, delta_s, delta_e, delta_h)
            
            # Dynamisches Limit: Je mehr Zeit vergangen, desto mehr Sprung erlaubt
            # Bei 40Hz Streaming: 25ms zwischen Befehlen → max 20°
            # Bei 500ms Pause (wegen Skips): max 20° + 500ms * 80°/s = 60°
            max_allowed = L.max_delta_per_cmd + time_since_last * 80.0  # 80°/s max Geschwindigkeit
            max_allowed = min(max_allowed, 90.0)  # Absolutes Maximum: 90°
            
            if max_delta > max_allowed:
                return False, (f"Sprung zu groß: {max_delta:.2f}° "
                             f"(max erlaubt: {max_allowed:.1f}° bei {time_since_last*1000:.0f}ms seit letztem Cmd)")
        
        # 3. Überhitzungsschutz
        if self._continuous_move_start is not None:
            elapsed = time.time() - self._continuous_move_start
            if elapsed > L.max_continuous_move_s:
                return False, (f"Überhitzungsschutz: {elapsed:.0f}s kontinuierliche "
                             f"Bewegung (max: {L.max_continuous_move_s}s)")
        
        return True, "OK"

    def validate_speed(self, spd: int, acc: int) -> tuple[bool, str]:
        """Prüft ob Geschwindigkeit/Beschleunigung sicher sind."""
        L = self.limits
        if spd > L.max_spd:
            return False, f"spd={spd} > max={L.max_spd}"
        if acc > L.max_acc:
            return False, f"acc={acc} > max={L.max_acc}"
        return True, "OK"
    
    def validate_read_position(self, pos: dict) -> tuple[bool, str]:
        """
        Prüft ob eine gelesene Position plausibel ist.
        DAS hätte den 30120°-Fehler verhindert!
        """
        if pos is None:
            return False, "Position ist None"
        
        L = self.limits
        for joint in ["b", "s", "e", "h"]:
            if joint not in pos:
                return False, f"Gelenk '{joint}' fehlt in Position"
            val = pos[joint]
            if abs(val) > L.max_plausible_position:
                return False, f"{joint}={val:.2f}° ist UNMÖGLICH (>{L.max_plausible_position}°)"
            if math.isnan(val) or math.isinf(val):
                return False, f"{joint} ist NaN/Inf"
        
        return True, "OK"
    
    def validate_error(self, error_deg: float) -> tuple[bool, str]:
        """
        Prüft ob ein berechneter Fehler plausibel ist.
        DAS hätte den 30120°-Fehler abgefangen!
        """
        if abs(error_deg) > self.limits.max_plausible_error:
            return False, (f"Fehler={error_deg:.2f}° ist unplausibel "
                         f"(max: {self.limits.max_plausible_error}°)")
        return True, "OK"
    
    def register_command(self, b: float, s: float, e: float, h: float):
        """Registriert einen gesendeten Befehl für Tracking."""
        now = time.time()
        
        if self._last_commanded is None:
            self._continuous_move_start = now
        
        self._last_commanded = {"b": b, "s": s, "e": e, "h": h, "t": now}
        self._total_commands_sent += 1
    
    def register_stream_start(self):
        """Markiert den Start einer Streaming-Session."""
        self._stream_start_time = time.time()
        self._continuous_move_start = time.time()
    
    def register_stream_end(self):
        """Markiert das Ende einer Streaming-Session."""
        self._stream_start_time = None
        self._continuous_move_start = None
    
    def register_pause(self):
        """Setzt den Überhitzungs-Timer zurück (z.B. nach einer Pause)."""
        self._continuous_move_start = None
    
    def trigger_emergency_stop(self, reason: str = ""):
        """Löst einen Software-Notaus aus."""
        self._emergency_stop = True
        print(f"\n{'!'*60}")
        print(f"  🚨 EMERGENCY STOP: {reason}")
        print(f"{'!'*60}\n")
    
    def reset_emergency_stop(self):
        """Setzt den Software-Notaus zurück (nur manuell!)."""
        self._emergency_stop = False
        self._last_commanded = None
        self._continuous_move_start = None
        print("  ✅ Emergency Stop zurückgesetzt")
    
    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stop
    
    def get_stats(self) -> dict:
        """Gibt Statistiken zurück."""
        elapsed = 0
        if self._continuous_move_start:
            elapsed = time.time() - self._continuous_move_start
        return {
            "total_commands": self._total_commands_sent,
            "continuous_move_s": elapsed,
            "emergency_stop": self._emergency_stop,
            "last_commanded": self._last_commanded,
        }


# ============================================================
# SAFE ARM WRAPPER
# ============================================================

class SafeArm:
    """
    Wrapper um RoArmConnection der ALLE Befehle durch Safety-Checks schickt.
    
    Verwendung:
        arm = RoArmConnection(port)
        safe_arm = SafeArm(arm)
        safe_arm.move_to(b, s, e, h, spd, acc)  # Wird geprüft!
    """
    
    def __init__(self, arm, limits: SafetyLimits = None, 
                 on_violation=None):
        """
        arm: RoArmConnection Instanz
        limits: SafetyLimits (oder Default)
        on_violation: Callback bei Verletzung: fn(reason: str)
        """
        self._arm = arm
        self._validator = PositionValidator(limits)
        self._on_violation = on_violation or self._default_violation_handler
        self._read_failures_consecutive = 0
        self._max_read_failures = 5
    
    def _default_violation_handler(self, reason: str):
        """Standard-Handler: Printet Warnung und stoppt."""
        print(f"\n  🛑 SAFETY VIOLATION: {reason}")
        print(f"     → Befehl NICHT gesendet!")
    
    def move_to(self, b: float, s: float, e: float, h: float,
                spd: int = 20, acc: int = 10) -> bool:
        """
        Sicherer Move. Gibt True zurück wenn gesendet, False wenn blockiert.
        """
        # Speed-Check
        ok, reason = self._validator.validate_speed(spd, acc)
        if not ok:
            self._on_violation(f"Speed: {reason}")
            return False
        
        # Position-Check
        ok, reason = self._validator.validate_target(b, s, e, h)
        if not ok:
            self._on_violation(f"Position: {reason}")
            # Bei Überhitzung: Emergency Stop
            if "Überhitzung" in reason:
                self._validator.trigger_emergency_stop(reason)
            return False
        
        # Alles OK → senden
        self._arm.move_to(b, s, e, h, spd=spd, acc=acc)
        self._validator.register_command(b, s, e, h)
        return True

    def trigger_emergency_stop(self, reason: str):
        self._validator.trigger_emergency_stop(reason)
        # Sofort sanft stoppen!
        GracefulStop.execute(self._arm, self._validator._last_commanded)
    
    def move_to_fast(self, b: float, s: float, e: float, h: float,
                     spd: int = 50, acc: int = 30) -> bool:
        """Sicherer Fast-Move (für Streaming)."""
        # Speed-Check
        ok, reason = self._validator.validate_speed(spd, acc)
        if not ok:
            self._on_violation(f"Speed: {reason}")
            return False
        
        # Position-Check
        ok, reason = self._validator.validate_target(b, s, e, h)
        if not ok:
            self._on_violation(f"Position: {reason}")
            if "Überhitzung" in reason or "Sprung" in reason:
                self._validator.trigger_emergency_stop(reason)
            return False
        
        # Senden
        self._arm.move_to_fast(b, s, e, h, spd=spd, acc=acc)
        self._validator.register_command(b, s, e, h)
        return True
    
    def read_position_deg(self) -> Optional[dict]:
        """
        Sicherer Position-Read mit Plausibilitätsprüfung.
        Gibt None zurück bei Müll-Daten.
        """
        pos = self._arm.read_position_deg()
        
        if pos is None:
            self._read_failures_consecutive += 1
            if self._read_failures_consecutive >= self._max_read_failures:
                self._validator.trigger_emergency_stop(
                    f"{self._max_read_failures}x hintereinander kein Read möglich!"
                )
            return None
        
        # Plausibilitätsprüfung
        ok, reason = self._validator.validate_read_position(pos)
        if not ok:
            self._read_failures_consecutive += 1
            print(f"\n  ⚠️ UNPLAUSIBLER READ: {reason}")
            print(f"     Rohdaten: {pos}")
            if self._read_failures_consecutive >= self._max_read_failures:
                self._validator.trigger_emergency_stop(
                    f"Zu viele unplausible Reads: {reason}"
                )
            return None
        
        # Alles OK
        self._read_failures_consecutive = 0
        return pos
    
    def safe_read_error(self, pos: dict, target: dict) -> Optional[float]:
        """
        Berechnet Fehler mit Plausibilitätsprüfung.
        DAS hätte den 30120°-Bug verhindert!
        """
        if pos is None:
            return None
        
        err = max(abs(pos[j] - target[j]) for j in ["b", "s", "e", "h"])
        
        ok, reason = self._validator.validate_error(err)
        if not ok:
            print(f"\n  🚨 UNPLAUSIBLER FEHLER: {reason}")
            print(f"     Pos:    {pos}")
            print(f"     Target: {target}")
            print(f"     → Ignoriere diesen Read!")
            return None
        
        return err
    
    def start_streaming(self):
        """Markiert Start einer Streaming-Session."""
        self._validator.register_stream_start()
        # Buffer leeren vor dem Streaming
        self._arm._ser.reset_input_buffer()
    
    def end_streaming(self):
        """Markiert Ende einer Streaming-Session + Buffer flush."""
        self._validator.register_stream_end()
        # WICHTIG: Buffer leeren nach Streaming!
        time.sleep(0.3)
        self._arm._ser.reset_input_buffer()
    
    def flush_and_read(self) -> Optional[dict]:
        """Flusht Buffer und liest dann sicher."""
        time.sleep(0.2)
        self._arm._ser.reset_input_buffer()
        time.sleep(0.1)
        return self.read_position_deg()
    
    # Durchreichen der anderen Methoden
    def torque_on(self):
        self._arm.torque_on()
    
    def torque_off(self):
        self._arm.torque_off()
        self._validator.register_pause()
    
    def gripper_open(self):
        self._arm.gripper_open()
    
    def gripper_close(self):
        self._arm.gripper_close()
    
    def close(self):
        self._arm.close()
    
    @property
    def is_emergency_stopped(self) -> bool:
        return self._validator.is_emergency_stopped
    
    def reset_emergency(self):
        self._validator.reset_emergency_stop()
    
    @property
    def validator(self) -> PositionValidator:
        return self._validator


# ============================================================
# WATCHDOG THREAD
# ============================================================

import threading

class SafetyWatchdog:
    """
    Hintergrund-Thread der den Arm überwacht.
    Löst Emergency Stop aus bei:
    - Zu langer Streaming-Dauer
    - Keine Antwort vom Arm
    """
    
    def __init__(self, safe_arm: SafeArm, check_interval: float = 2.0):
        self._safe_arm = safe_arm
        self._check_interval = check_interval
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        print("   🐕 Safety Watchdog gestartet")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
    
    def _watchdog_loop(self):
        while self._running:
            time.sleep(self._check_interval)
            
            if self._safe_arm.is_emergency_stopped:
                continue
            
            stats = self._safe_arm.validator.get_stats()
            
            # Check: Zu lange kontinuierliche Bewegung
            if stats["continuous_move_s"] > self._safe_arm.validator.limits.max_continuous_move_s:
                self._safe_arm.validator.trigger_emergency_stop(
                    f"Überhitzungsschutz: {stats['continuous_move_s']:.0f}s "
                    f"kontinuierliche Bewegung!"
                )

# In safety.py ergänzen:

class CurrentMonitor:
    """
    Überwacht den Servo-Strom.
    Hoher Strom + keine Bewegung = Servo drückt gegen Anschlag/Hindernis.
    """

    def __init__(self, safe_arm, max_load_percent: float = 85.0,
                 max_stall_duration_s: float = 2.0):
        self._arm = safe_arm
        self._max_load = max_load_percent
        self._max_stall_s = max_stall_duration_s
        self._stall_start = None
        self._last_position = None
        self._position_unchanged_since = None

    def check(self, current_pos: dict, load_percent: float = None) -> tuple[bool, str]:
        """
        Prüft ob der Arm in einem gefährlichen Zustand ist.
        Returns: (is_safe, reason)
        """
        now = time.time()

        # 1. Stall-Erkennung: Position ändert sich nicht trotz aktiver Befehle
        if self._last_position is not None and current_pos is not None:
            max_delta = max(abs(current_pos[j] - self._last_position[j])
                          for j in ["b", "s", "e", "h"])

            if max_delta < 0.1:  # Weniger als 0.1° Änderung
                if self._position_unchanged_since is None:
                    self._position_unchanged_since = now
                elif now - self._position_unchanged_since > self._max_stall_s:
                    return False, (f"STALL ERKANNT: Position unverändert seit "
                                  f"{now - self._position_unchanged_since:.1f}s "
                                  f"trotz aktiver Befehle!")
            else:
                self._position_unchanged_since = None

        self._last_position = current_pos

        # 2. Überstrom (wenn Servo-Load auslesbar)
        if load_percent is not None and load_percent > self._max_load:
            if self._stall_start is None:
                self._stall_start = now
            elif now - self._stall_start > 1.0:  # >1s Überlast
                return False, (f"ÜBERLAST: {load_percent:.0f}% > {self._max_load:.0f}% "
                              f"seit {now - self._stall_start:.1f}s")
        else:
            self._stall_start = None

        return True, "OK"

    def reset(self):
        self._stall_start = None
        self._last_position = None
        self._position_unchanged_since = None

# In play.py, nach dem Laden der Trajektorie:

class TrajectoryValidator:
    """Prüft die gesamte Trajektorie VOR dem Abspielen."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits

    def validate_full_trajectory(self, trajectory,
                                  hz: int = 100) -> tuple[bool, list]:
        """
        Samplet die gesamte Trajektorie mit hoher Rate und prüft:
        - Alle Positionen innerhalb der Grenzen
        - Keine zu großen Sprünge zwischen Samples
        - Geschwindigkeit pro Gelenk nie über Maximum
        - Beschleunigung pro Gelenk nie über Maximum

        Returns: (is_safe, list_of_violations)
        """
        duration = trajectory.get_duration()
        n_samples = int(duration * hz)
        dt = 1.0 / hz
        violations = []

        L = self.limits
        MAX_JOINT_VELOCITY_DEG_S = 180.0   # Max 180°/s pro Gelenk
        MAX_JOINT_ACCEL_DEG_S2 = 500.0     # Max 500°/s² pro Gelenk

        prev_pos = None
        prev_vel = None

        for i in range(n_samples):
            t = i * dt
            pos = trajectory.sample(t)

            # 1. Absolute Grenzen
            if not (L.b_min <= pos["b"] <= L.b_max):
                violations.append(f"t={t:.3f}s: b={pos['b']:.2f}° außerhalb Grenzen")
            if not (L.s_min <= pos["s"] <= L.s_max):
                violations.append(f"t={t:.3f}s: s={pos['s']:.2f}° außerhalb Grenzen")
            if not (L.e_min <= pos["e"] <= L.e_max):
                violations.append(f"t={t:.3f}s: e={pos['e']:.2f}° außerhalb Grenzen")
            if not (L.h_min <= pos["h"] <= L.h_max):
                violations.append(f"t={t:.3f}s: h={pos['h']:.2f}° außerhalb Grenzen")

            # 2. Geschwindigkeit (°/s)
            if prev_pos is not None:
                vel = {j: (pos[j] - prev_pos[j]) / dt for j in ["b", "s", "e", "h"]}
                for j in ["b", "s", "e", "h"]:
                    if abs(vel[j]) > MAX_JOINT_VELOCITY_DEG_S:
                        violations.append(
                            f"t={t:.3f}s: {j} Geschwindigkeit={vel[j]:.1f}°/s "
                            f"> max {MAX_JOINT_VELOCITY_DEG_S}°/s")

                # 3. Beschleunigung (°/s²)
                if prev_vel is not None:
                    for j in ["b", "s", "e", "h"]:
                        accel = (vel[j] - prev_vel[j]) / dt
                        if abs(accel) > MAX_JOINT_ACCEL_DEG_S2:
                            violations.append(
                                f"t={t:.3f}s: {j} Beschleunigung={accel:.0f}°/s² "
                                f"> max {MAX_JOINT_ACCEL_DEG_S2}°/s²")

                prev_vel = vel

            prev_pos = pos

            # Abbruch bei zu vielen Violations (Performance)
            if len(violations) > 20:
                violations.append("... (abgebrochen, zu viele Fehler)")
                break

        is_safe = len(violations) == 0
        return is_safe, violations

# In safety.py ergänzen:

class GracefulStop:
    """
    Bei Emergency: Nicht einfach aufhören, sondern den Arm
    sanft zur aktuellen Position bremsen und dann Torque reduzieren.
    """

    @staticmethod
    def execute(arm_raw, last_known_pos: dict = None):
        """
        Führt einen sanften Stopp durch:
        1. Aktuellen Befehl mit sehr niedriger Geschwindigkeit zur
           aktuellen Position senden (= "bleib wo du bist")
        2. Kurz warten
        3. Torque aus (Arm wird schlaff → kein Druck mehr)
        """
        print("\n  🔄 Graceful Stop wird ausgeführt...")

        try:
            # Schritt 1: Position lesen (falls möglich)
            if last_known_pos is None:
                time.sleep(0.2)
                arm_raw._ser.reset_input_buffer()
                time.sleep(0.1)
                pos = arm_raw.read_position_deg()
                if pos:
                    last_known_pos = pos

            # Schritt 2: "Bleib wo du bist" mit minimaler Kraft
            if last_known_pos:
                arm_raw.move_to(
                    last_known_pos["b"], last_known_pos["s"],
                    last_known_pos["e"], last_known_pos["h"],
                    spd=5, acc=2  # Sehr langsam = wenig Kraft
                )
                time.sleep(0.5)

            # Schritt 3: Torque aus → Arm wird schlaff
            # Das ist der sicherste Zustand: kein Strom, kein Druck
            arm_raw.torque_off()
            print("  ✅ Torque AUS - Arm ist schlaff (sicher)")

        except Exception as e:
            # Im Notfall: Einfach Torque aus
            try:
                arm_raw.torque_off()
            except:
                pass
            print(f"  ⚠️  Graceful Stop Fehler: {e}, Torque wurde ausgeschaltet")

class ThermalEstimator:
    """
    Schätzt die Servo-Temperatur basierend auf:
    - Wie lange Befehle gesendet werden (= Strom fließt)
    - Wie groß die Bewegungen sind (= mehr Arbeit = mehr Wärme)
    - Pausen (= Abkühlung)

    Einfaches thermisches Modell:
    T_servo = T_ambient + integral(power) * thermal_resistance
    """

    def __init__(self, ambient_temp_c: float = 25.0,
                 thermal_time_constant_s: float = 120.0,
                 max_safe_temp_c: float = 55.0,
                 warning_temp_c: float = 45.0):
        self._ambient = ambient_temp_c
        self._tau = thermal_time_constant_s  # Wie schnell kühlt der Servo ab
        self._max_temp = max_safe_temp_c
        self._warn_temp = warning_temp_c
        self._estimated_temp = ambient_temp_c
        self._last_update = time.time()
        self._is_active = False  # Ob gerade Befehle gesendet werden
        self._power_level = 0.0  # 0-1, geschätzte Last

    def update(self, is_moving: bool = False, delta_deg: float = 0.0):
        """Wird bei jedem Befehl aufgerufen."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Geschätzte Leistung (vereinfacht)
        if is_moving:
            # Mehr Bewegung = mehr Strom = mehr Wärme
            self._power_level = min(1.0, delta_deg / 10.0)  # 10°/Befehl = volle Last
        else:
            self._power_level = 0.1  # Haltestrom (Torque on, keine Bewegung)

        # Thermisches Modell: Aufheizen/Abkühlen
        # dT/dt = (P * R_th - (T - T_amb)) / tau
        max_temp_rise = 40.0  # Max 40°C über Ambient bei Volllast
        target_temp = self._ambient + self._power_level * max_temp_rise

        # Exponentielles Annähern an target_temp
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._estimated_temp += alpha * (target_temp - self._estimated_temp)

    def update_idle(self):
        """Wird aufgerufen wenn KEINE Befehle gesendet werden (Abkühlung)."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Abkühlung Richtung Ambient
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._estimated_temp += alpha * (self._ambient - self._estimated_temp)

    def get_status(self) -> tuple[float, str]:
        """
        Returns: (estimated_temp_c, status)
        status: "OK", "WARM", "HOT", "CRITICAL"
        """
        t = self._estimated_temp
        if t >= self._max_temp:
            return t, "CRITICAL"
        elif t >= self._warn_temp:
            return t, "HOT"
        elif t >= self._ambient + 10:
            return t, "WARM"
        return t, "OK"

    def should_pause(self) -> bool:
        """True wenn eine Zwangspause nötig ist."""
        return self._estimated_temp >= self._max_temp

    def get_recommended_pause_s(self) -> float:
        """Wie lange sollte pausiert werden um auf sichere Temperatur zu kommen."""
        if self._estimated_temp <= self._warn_temp:
            return 0.0
        # Wie lange dauert es bis wir auf warn_temp abgekühlt sind?
        # T(t) = T_amb + (T_now - T_amb) * exp(-t/tau)
        # warn = amb + (now - amb) * exp(-t/tau)
        # exp(-t/tau) = (warn - amb) / (now - amb)
        ratio = (self._warn_temp - self._ambient) / max(
            self._estimated_temp - self._ambient, 0.1)
        if ratio <= 0 or ratio >= 1:
            return 30.0  # Fallback
        return -self._tau * math.log(ratio)

class RateLimiter:
    """Stellt sicher dass nie mehr als max_hz Befehle/s gesendet werden."""

    def __init__(self, max_hz: float = 60.0):
        self._min_interval = 1.0 / max_hz
        self._last_send_time = 0.0
        self._violations = 0

    def acquire(self) -> bool:
        """
        Gibt True zurück wenn gesendet werden darf.
        Wartet ggf. kurz.
        """
        now = time.time()
        elapsed = now - self._last_send_time

        if elapsed < self._min_interval:
            # Zu schnell! Kurz warten.
            wait = self._min_interval - elapsed
            time.sleep(wait)
            self._violations += 1

        self._last_send_time = time.time()
        return True

    @property
    def violations(self) -> int:
        return self._violations
