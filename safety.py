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
        
        # 2. Sprung-Erkennung (zu große Änderung auf einmal)
        if self._last_commanded is not None:
            delta_b = abs(b - self._last_commanded["b"])
            delta_s = abs(s - self._last_commanded["s"])
            delta_e = abs(e - self._last_commanded["e"])
            delta_h = abs(h - self._last_commanded["h"])
            max_delta = max(delta_b, delta_s, delta_e, delta_h)
            
            if max_delta > L.max_delta_per_cmd:
                return False, (f"Sprung zu groß: {max_delta:.2f}° "
                             f"(max erlaubt: {L.max_delta_per_cmd}°)")
        
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
        print("  🐕 Safety Watchdog gestartet")
    
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
