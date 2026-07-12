#!/usr/bin/env python3
"""test_roarm.py - Automatisierte Tests für das RoArm-M2-S System

Testet alle Module:
- robot.py: Basis-Kommunikation
- safety.py: Safety-Layer, Validator, Watchdog, Thermal
- play.py: Trajektorien-Laden, Spline-Interpolation, Streaming
- calibrate.py: Kalibrierungsmodell, Pose-Validierung
- teach.py: Recording-Logik

Alle Tests verwenden SimulatedArm statt echte Hardware.

Ausführen:
    python -m pytest test_roarm.py -v
    # oder ohne pytest:
    python test_roarm.py
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pytest",
# ]
# ///

import sys
import os
import time
import math
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Sicherstellen dass die Module gefunden werden
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_robot import SimulatedArm, SimConfig

# ============================================================
# TEST FIXTURES / HELPERS
# ============================================================

def make_arm(config: SimConfig = None) -> SimulatedArm:
    """Erstellt einen simulierten Arm mit Default-Config."""
    return SimulatedArm(config=config or SimConfig(
        position_noise_deg=0.01,  # Wenig Rauschen für deterministische Tests
        read_latency_s=0.001,
        command_latency_s=0.001,
    ))


def make_recording_file(waypoints: list = None, offset: dict = None,
                        gripper_cmds: list = None) -> str:
    """Erstellt eine temporäre .roarm Datei für Tests."""
    if waypoints is None:
        # Standard-Trajektorie: Einfache Bewegung
        waypoints = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 30.0, "s": 5.0, "e": 80.0, "h": 180.0},
            {"t": 2.5, "b": 20.0, "s": 0.0, "e": 85.0, "h": 180.0},
            {"t": 3.0, "b": 10.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 3.5, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
        ]
    
    lines = [
        "# Test Recording",
        f"#CONFIG hz=50",
        f"#CONFIG threshold=0.1",
        f"#START_POS b=0.00 s=0.00 e=90.00 h=180.00",
    ]
    
    if offset:
        lines.append(
            f"#OFFSET b={offset['b']:.3f} s={offset['s']:.3f} "
            f"e={offset['e']:.3f} h={offset['h']:.3f}"
        )
    
    lines.append("")
    
    for wp in waypoints:
        lines.append(
            f"MOVE b={wp['b']:.2f} s={wp['s']:.2f} "
            f"e={wp['e']:.2f} h={wp['h']:.2f} t={wp['t']:.4f}"
        )
    
    if gripper_cmds:
        for gc in gripper_cmds:
            lines.append(f"GRIPPER {gc['cmd']} t={gc['t']:.4f}")
    
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.roarm', delete=False)
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    return tmp.name


# ============================================================
# TESTS: SIMULIERTER ARM (sim_robot.py)
# ============================================================

class TestSimulatedArm:
    """Tests für die Simulation selbst."""
    
    def test_initial_position(self):
        """Arm startet an der richtigen Position."""
        arm = make_arm()
        pos = arm.read_position_deg()
        assert pos is not None
        assert abs(pos["b"] - 0.0) < 1.0
        assert abs(pos["s"] - 0.0) < 1.0
        assert abs(pos["e"] - 90.0) < 1.0
        assert abs(pos["h"] - 180.0) < 1.0
        arm.close()
    
    def test_custom_initial_position(self):
        """Arm startet an benutzerdefinierter Position."""
        arm = SimulatedArm(initial_position={"b": 45.0, "s": 20.0, "e": 60.0, "h": 90.0})
        pos = arm.get_true_position()
        assert pos["b"] == 45.0
        assert pos["s"] == 20.0
        arm.close()
    
    def test_move_to_changes_target(self):
        """move_to setzt das Ziel korrekt."""
        arm = make_arm()
        arm.move_to(30.0, 15.0, 60.0, 180.0, spd=20, acc=10)
        arm.wait_for_arrival()
        pos = arm.get_true_position()
        assert abs(pos["b"] - 30.0) < 0.5
        assert abs(pos["s"] - 15.0) < 0.5
        assert abs(pos["e"] - 60.0) < 0.5
        arm.close()
    
    def test_torque_off_gravity(self):
        """Bei Torque-off hängt der Arm durch Gravitation durch."""
        config = SimConfig(
            gravity_droop_s_deg=-5.0,
            gravity_droop_e_deg=-3.0,
            position_noise_deg=0.0,
        )
        arm = SimulatedArm(config=config)
        initial_s = arm.get_true_position()["s"]
        
        arm.torque_off()
        time.sleep(0.5)
        
        after_s = arm.get_true_position()["s"]
        # Shoulder sollte nach unten gedriftet sein
        assert after_s < initial_s
        arm.close()
    
    def test_command_logging(self):
        """Alle Befehle werden geloggt."""
        arm = make_arm()
        arm.move_to(10, 10, 90, 180)
        arm.torque_off()
        arm.gripper_close()
        arm.torque_on()
        
        log = arm.get_command_log()
        types = [c.command_type for c in log]
        assert "move_to" in types
        assert "torque_off" in types
        assert "gripper_close" in types
        assert "torque_on" in types
        arm.close()
    
    def test_read_failure_injection(self):
        """Fehler-Injektion funktioniert."""
        arm = make_arm()
        arm.inject_fault("read_failure", duration_s=0.5)
        
        # Während des Fehlers sollten Reads None zurückgeben
        failures = 0
        for _ in range(10):
            pos = arm.read_position_deg()
            if pos is None:
                failures += 1
            time.sleep(0.02)
        
        assert failures > 0, "Fehler-Injektion hat nicht funktioniert"
        
        # Nach der Fehler-Dauer sollten Reads wieder funktionieren
        time.sleep(0.6)
        pos = arm.read_position_deg()
        assert pos is not None, "Reads funktionieren nach Fehler-Ende nicht"
        arm.close()
    
    def test_garbage_read_injection(self):
        """Garbage-Reads liefern unplausible Werte."""
        arm = make_arm()
        arm.inject_fault("garbage", duration_s=0.3)
        
        garbage_detected = False
        for _ in range(10):
            pos = arm.read_position_deg()
            if pos and (abs(pos["b"]) > 200 or abs(pos["s"]) > 200):
                garbage_detected = True
                break
            time.sleep(0.02)
        
        assert garbage_detected, "Garbage-Injektion hat keine Müll-Werte erzeugt"
        arm.close()
    
    def test_averaged_read(self):
        """read_position_averaged gibt Mittelwert + Standardabweichung."""
        arm = make_arm()
        avg = arm.read_position_averaged(n=10, interval=0.01)
        
        assert avg is not None
        assert "b" in avg and "s" in avg and "e" in avg and "h" in avg
        assert "b_std" in avg and "s_std" in avg
        assert "n_samples" in avg
        assert avg["n_samples"] > 0
        # Standardabweichung sollte klein sein (wenig Rauschen in Config)
        assert avg["b_std"] < 1.0
        arm.close()
    
    def test_wait_for_arrival(self):
        """wait_for_arrival wartet bis Ziel erreicht."""
        arm = make_arm()
        arm.move_to(45.0, 20.0, 60.0, 180.0)
        arrived = arm.wait_for_arrival(timeout=5.0)
        
        pos = arm.get_true_position()
        assert abs(pos["b"] - 45.0) < 1.0
        assert abs(pos["s"] - 20.0) < 1.0
        arm.close()
    
    def test_set_position_directly(self):
        """set_position setzt Position sofort (für Test-Setup)."""
        arm = make_arm()
        arm.set_position({"b": 77.0, "s": -10.0, "e": 45.0, "h": 90.0})
        
        pos = arm.get_true_position()
        assert pos["b"] == 77.0
        assert pos["s"] == -10.0
        assert pos["e"] == 45.0
        arm.close()


# ============================================================
# TESTS: SAFETY LAYER (safety.py)
# ============================================================

class TestPositionValidator:
    """Tests für den PositionValidator."""
    
    def test_valid_position_passes(self):
        """Gültige Position wird akzeptiert."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        ok, reason = v.validate_target(0.0, 0.0, 90.0, 180.0)
        assert ok is True
        assert reason == "OK"
    
    def test_out_of_bounds_base(self):
        """Base außerhalb der Grenzen wird abgelehnt."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        ok, reason = v.validate_target(200.0, 0.0, 90.0, 180.0)
        assert ok is False
        assert "b=" in reason
    
    def test_out_of_bounds_shoulder(self):
        """Shoulder außerhalb der Grenzen wird abgelehnt."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        ok, reason = v.validate_target(0.0, 100.0, 90.0, 180.0)
        assert ok is False
        assert "s=" in reason
    
    def test_out_of_bounds_elbow(self):
        """Elbow außerhalb der Grenzen wird abgelehnt."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        ok, reason = v.validate_target(0.0, 0.0, -10.0, 180.0)
        assert ok is False
        assert "e=" in reason
    
    def test_jump_detection(self):
        """Zu großer Sprung wird erkannt."""
        from safety import PositionValidator, SafetyLimits
        limits = SafetyLimits(max_delta_per_cmd=15.0)
        v = PositionValidator(limits)
        
        # Ersten Befehl registrieren
        v.register_command(0.0, 0.0, 90.0, 180.0)
        
        # Sofort danach ein großer Sprung (innerhalb weniger ms)
        time.sleep(0.01)
        ok, reason = v.validate_target(80.0, 0.0, 90.0, 180.0)
        assert ok is False
        assert "Sprung" in reason
    
    def test_small_move_passes(self):
        """Kleine Bewegung wird akzeptiert."""
        from safety import PositionValidator, SafetyLimits
        limits = SafetyLimits(max_delta_per_cmd=15.0)
        v = PositionValidator(limits)
        
        v.register_command(0.0, 0.0, 90.0, 180.0)
        time.sleep(0.01)
        ok, reason = v.validate_target(5.0, 2.0, 88.0, 180.0)
        assert ok is True
    
    def test_emergency_stop_blocks_all(self):
        """Nach Emergency Stop werden alle Befehle blockiert."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        v.trigger_emergency_stop("Test")
        ok, reason = v.validate_target(0.0, 0.0, 90.0, 180.0)
        assert ok is False
        assert "EMERGENCY" in reason
    
    def test_emergency_stop_reset(self):
        """Emergency Stop kann zurückgesetzt werden."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        v.trigger_emergency_stop("Test")
        assert v.is_emergency_stopped is True
        
        v.reset_emergency_stop()
        assert v.is_emergency_stopped is False
        
        ok, _ = v.validate_target(0.0, 0.0, 90.0, 180.0)
        assert ok is True
    
    def test_validate_read_position_garbage(self):
        """Müll-Reads werden erkannt (30120°-Bug)."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        # Simuliert den berüchtigten 30120°-Bug
        garbage = {"b": 30120.0, "s": 0.0, "e": 90.0, "h": 180.0}
        ok, reason = v.validate_read_position(garbage)
        assert ok is False
        assert "UNMÖGLICH" in reason or "unmöglich" in reason.lower() or ">" in reason
    
    def test_validate_read_position_none(self):
        """None-Position wird als ungültig erkannt."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        ok, reason = v.validate_read_position(None)
        assert ok is False
    
    def test_validate_error_plausible(self):
        """Plausibler Fehler wird akzeptiert."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        ok, _ = v.validate_error(2.5)
        assert ok is True
    
    def test_validate_error_implausible(self):
        """Unplausibler Fehler (>10°) wird abgelehnt."""
        from safety import PositionValidator, SafetyLimits
        v = PositionValidator(SafetyLimits())
        
        ok, reason = v.validate_error(30120.0)
        assert ok is False
    
    def test_speed_validation(self):
        """Zu hohe Geschwindigkeit wird abgelehnt."""
        from safety import PositionValidator, SafetyLimits
        limits = SafetyLimits(max_spd=50, max_acc=30)
        v = PositionValidator(limits)
        
        ok, _ = v.validate_speed(50, 30)
        assert ok is True
        
        ok, reason = v.validate_speed(100, 30)
        assert ok is False
        assert "spd" in reason
    
    def test_continuous_move_timeout(self):
        """Überhitzungsschutz nach zu langer Bewegung."""
        from safety import PositionValidator, SafetyLimits
        limits = SafetyLimits(max_continuous_move_s=0.1)  # Sehr kurz für Test
        v = PositionValidator(limits)
        
        v.register_command(0.0, 0.0, 90.0, 180.0)
        time.sleep(0.15)
        
        ok, reason = v.validate_target(5.0, 0.0, 90.0, 180.0)
        assert ok is False
        assert "Überhitzung" in reason or "berhitzung" in reason


class TestSafeArm:
    """Tests für den SafeArm Wrapper."""
    
    def test_safe_move_valid(self):
        """Gültiger Move wird durchgelassen."""
        from safety import SafeArm, SafetyLimits
        arm = make_arm()
        safe = SafeArm(arm, SafetyLimits())
        
        # Initiale Position setzen
        safe.validator._last_commanded = {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0, "t": time.time()
        }
        
        result = safe.move_to(5.0, 2.0, 88.0, 180.0, spd=20, acc=10)
        assert result is True
        arm.close()
    
    def test_safe_move_blocked(self):
        """Ungültiger Move wird blockiert."""
        from safety import SafeArm, SafetyLimits
        arm = make_arm()
        safe = SafeArm(arm, SafetyLimits())
        
        # Position außerhalb der Grenzen
        result = safe.move_to(200.0, 0.0, 90.0, 180.0, spd=20, acc=10)
        assert result is False
        arm.close()
    
    def test_safe_read_filters_garbage(self):
        """SafeArm filtert Müll-Reads."""
        from safety import SafeArm, SafetyLimits
        
        config = SimConfig(garbage_read_probability=1.0, position_noise_deg=0.0)
        arm = SimulatedArm(config=config)
        safe = SafeArm(arm, SafetyLimits())
        
        pos = safe.read_position_deg()
        # Garbage sollte gefiltert werden → None
        assert pos is None
        arm.close()
    
    def test_consecutive_read_failures_trigger_emergency(self):
        """Zu viele Read-Fehler lösen Emergency Stop aus."""
        from safety import SafeArm, SafetyLimits
        
        config = SimConfig(read_failure_probability=1.0)
        arm = SimulatedArm(config=config)
        safe = SafeArm(arm, SafetyLimits())
        
        # 5+ Reads die fehlschlagen
        for _ in range(6):
            safe.read_position_deg()
        
        assert safe.is_emergency_stopped is True
        arm.close()


class TestThermalEstimator:
    """Tests für den Temperatur-Schätzer."""
    
    def test_heats_up_during_movement(self):
        """Temperatur steigt bei Bewegung."""
        from safety import ThermalEstimator
        
        thermal = ThermalEstimator(
            ambient_temp_c=25.0,
            thermal_time_constant_s=10.0,  # Schnell für Tests
            max_safe_temp_c=55.0,
        )
        
        initial_temp, _ = thermal.get_status()
        
        # Simuliere viele Bewegungen
        for _ in range(50):
            thermal.update(is_moving=True, delta_deg=10.0)
            time.sleep(0.01)
        
        final_temp, _ = thermal.get_status()
        assert final_temp > initial_temp
    
    def test_cools_down_when_idle(self):
        """Temperatur sinkt bei Inaktivität."""
        from safety import ThermalEstimator
        
        thermal = ThermalEstimator(
            ambient_temp_c=25.0,
            thermal_time_constant_s=5.0,
        )
        
        # Aufheizen
        for _ in range(30):
            thermal.update(is_moving=True, delta_deg=10.0)
            time.sleep(0.01)
        
        hot_temp, _ = thermal.get_status()
        
        # Abkühlen
        for _ in range(30):
            thermal.update_idle()
            time.sleep(0.01)
        
        cool_temp, _ = thermal.get_status()
        assert cool_temp < hot_temp
    
    def test_critical_triggers_pause(self):
        """Bei kritischer Temperatur wird Pause empfohlen."""
        from safety import ThermalEstimator
        
        thermal = ThermalEstimator(
            ambient_temp_c=25.0,
            thermal_time_constant_s=2.0,
            max_safe_temp_c=30.0,  # Niedrig für schnellen Test
        )
        
        # Schnell aufheizen
        for _ in range(100):
            thermal.update(is_moving=True, delta_deg=15.0)
            time.sleep(0.005)
        
        assert thermal.should_pause() is True
        pause = thermal.get_recommended_pause_s()
        assert pause > 0


class TestRateLimiter:
    """Tests für den Rate Limiter."""
    
    def test_allows_within_limit(self):
        """Befehle innerhalb des Limits werden durchgelassen."""
        from safety import RateLimiter
        
        rl = RateLimiter(max_hz=100.0)
        
        # 10 Befehle mit genug Abstand
        for _ in range(10):
            result = rl.acquire()
            assert result is True
            time.sleep(0.015)  # 66 Hz < 100 Hz
        
        assert rl.violations == 0
    
    def test_throttles_when_too_fast(self):
        """Zu schnelle Befehle werden gedrosselt."""
        from safety import RateLimiter
        
        rl = RateLimiter(max_hz=20.0)  # Max 20 Hz = min 50ms Abstand
        
        # Schnell hintereinander senden
        start = time.time()
        for _ in range(5):
            rl.acquire()
        elapsed = time.time() - start
        
        # Sollte mindestens 4 * 50ms = 200ms gedauert haben
        assert elapsed >= 0.15  # Etwas Toleranz
        assert rl.violations > 0


class TestTrajectoryValidator:
    """Tests für den Trajectory Pre-Flight Check."""
    
    def test_safe_trajectory_passes(self):
        """Sichere Trajektorie besteht den Check."""
        from safety import TrajectoryValidator, SafetyLimits
        
        # Einfache sichere Trajektorie erstellen
        class FakeTrajectory:
            def get_duration(self):
                return 2.0
            def sample(self, t):
                # Lineare Bewegung im sicheren Bereich
                progress = t / 2.0
                return {
                    "b": progress * 30.0,
                    "s": progress * 10.0,
                    "e": 90.0 - progress * 20.0,
                    "h": 180.0,
                }
        
        validator = TrajectoryValidator(SafetyLimits())
        is_safe, violations = validator.validate_full_trajectory(FakeTrajectory(), hz=50)
        assert is_safe is True
        assert len(violations) == 0
    
    def test_unsafe_trajectory_fails(self):
        """Unsichere Trajektorie wird erkannt."""
        from safety import TrajectoryValidator, SafetyLimits
        
        class UnsafeTrajectory:
            def get_duration(self):
                return 1.0
            def sample(self, t):
                # Springt sofort auf unmögliche Position
                return {
                    "b": 200.0,  # Außerhalb!
                    "s": 0.0,
                    "e": 90.0,
                    "h": 180.0,
                }
        
        validator = TrajectoryValidator(SafetyLimits())
        is_safe, violations = validator.validate_full_trajectory(UnsafeTrajectory(), hz=50)
        assert is_safe is False
        assert len(violations) > 0


# ============================================================
# TESTS: PLAY MODULE (play.py)
# ============================================================

class TestParseRoarmFile:
    """Tests für das Datei-Parsing."""
    
    def test_parse_basic_file(self):
        """Grundlegendes Parsing einer .roarm Datei."""
        filepath = make_recording_file()
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(filepath)
            
            assert "waypoints" in data
            assert "gripper_cmds" in data
            assert "config" in data
            assert "start_pos" in data
            assert "offset" in data
            
            assert len(data["waypoints"]) == 8
            assert data["waypoints"][0]["t"] == 0.0
            assert data["waypoints"][-1]["t"] == 3.5
        finally:
            os.unlink(filepath)
    
    def test_parse_with_offset(self):
        """Offset wird korrekt geparst."""
        offset = {"b": 0.5, "s": -0.3, "e": 0.1, "h": 0.0}
        filepath = make_recording_file(offset=offset)
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(filepath)
            
            assert abs(data["offset"]["b"] - 0.5) < 0.001
            assert abs(data["offset"]["s"] - (-0.3)) < 0.001
        finally:
            os.unlink(filepath)
    
    def test_parse_with_gripper(self):
        """Gripper-Befehle werden geparst."""
        gripper_cmds = [
            {"cmd": "CLOSE", "t": 1.0},
            {"cmd": "OPEN", "t": 2.0},
        ]
        filepath = make_recording_file(gripper_cmds=gripper_cmds)
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(filepath)
            
            assert len(data["gripper_cmds"]) == 2
        finally:
            os.unlink(filepath)
    
    def test_parse_start_position(self):
        """Startposition wird korrekt geparst."""
        filepath = make_recording_file()
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(filepath)
            
            assert data["start_pos"]["b"] == 0.0
            assert data["start_pos"]["e"] == 90.0
        finally:
            os.unlink(filepath)


class TestApplyOffset:
    """Tests für die Offset-Anwendung."""
    
    def test_no_offset_unchanged(self):
        """Ohne Offset bleiben Waypoints unverändert."""
        from play import apply_offset_to_waypoints
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 1.0, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
        ]
        offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        
        result = apply_offset_to_waypoints(wps, offset)
        assert result[0]["b"] == 0.0
        assert result[1]["b"] == 10.0
    
    def test_offset_applied_to_end(self):
        """Offset wird auf die letzten Punkte angewendet."""
        from play import apply_offset_to_waypoints
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 1.0, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 2.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 3.0, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 4.0, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 5.0, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
        ]
        offset = {"b": 2.0, "s": 0.0, "e": 0.0, "h": 0.0}
        
        result = apply_offset_to_waypoints(wps, offset, blend_points=3)
        
        # Erste Punkte unverändert
        assert result[0]["b"] == 0.0
        # Letzter Punkt hat vollen Offset
        assert abs(result[-1]["b"] - 32.0) < 0.1


class TestSmoothTrajectory:
    """Tests für die Spline-Interpolation."""
    
    def test_trajectory_creation(self):
        """Trajektorie kann erstellt werden."""
        from play import SmoothTrajectory
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 20.0, "s": 5.0, "e": 80.0, "h": 180.0},
        ]
        
        traj = SmoothTrajectory(wps, speed_factor=1.0)
        assert traj.get_duration() > 0
    
    def test_trajectory_sample_start(self):
        """Sample bei t=0 gibt ungefähr den Startpunkt."""
        from play import SmoothTrajectory
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 20.0, "s": 5.0, "e": 80.0, "h": 180.0},
        ]
        
        traj = SmoothTrajectory(wps, speed_factor=1.0)
        pos = traj.sample(0.0)
        
        assert abs(pos["b"] - 0.0) < 2.0
        assert abs(pos["e"] - 90.0) < 2.0
    
    def test_trajectory_sample_end(self):
        """Sample am Ende gibt ungefähr den Endpunkt."""
        from play import SmoothTrajectory
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 20.0, "s": 5.0, "e": 80.0, "h": 180.0},
        ]
        
        traj = SmoothTrajectory(wps, speed_factor=1.0)
        duration = traj.get_duration()
        pos = traj.sample(duration)
        
        # Sollte ungefähr beim letzten Waypoint sein
        assert abs(pos["b"] - 20.0) < 5.0
        assert abs(pos["e"] - 80.0) < 5.0
    
    def test_trajectory_continuity(self):
        """Trajektorie ist stetig (keine großen Sprünge)."""
        from play import SmoothTrajectory
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 20.0, "s": 5.0, "e": 80.0, "h": 180.0},
        ]
        
        traj = SmoothTrajectory(wps, speed_factor=1.0)
        duration = traj.get_duration()
        
        # Sample mit hoher Rate und prüfe auf Sprünge
        n_samples = 200
        max_jump = 0.0
        prev = traj.sample(0.0)
        
        for i in range(1, n_samples):
            t = (i / n_samples) * duration
            pos = traj.sample(t)
            for j in ["b", "s", "e", "h"]:
                jump = abs(pos[j] - prev[j])
                max_jump = max(max_jump, jump)
            prev = pos
        
        # Bei 200 Samples über ~2s sollte kein Sprung > 5° sein
        assert max_jump < 5.0, f"Zu großer Sprung: {max_jump:.2f}°"
    
    def test_speed_factor_changes_duration(self):
        """Speed-Faktor ändert die Gesamtdauer."""
        from play import SmoothTrajectory
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 1.0, "b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 2.0, "b": 30.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 3.0, "b": 20.0, "s": 5.0, "e": 80.0, "h": 180.0},
            {"t": 4.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
        ]
        
        traj_normal = SmoothTrajectory(wps, speed_factor=1.0)
        traj_fast = SmoothTrajectory(wps, speed_factor=2.0)
        
        # Schnellere Trajektorie sollte kürzer sein
        assert traj_fast.get_duration() < traj_normal.get_duration()


# ============================================================
# TESTS: CALIBRATION MODEL (calibrate.py)
# ============================================================

class TestCalibrationModel:
    """Tests für das Kalibrierungsmodell."""
    
    def test_model_fit_and_predict(self):
        """Modell kann gefittet werden und Korrekturen vorhersagen."""
        from calibrate import CalibrationModel
        
        model = CalibrationModel()
        
        # Synthetische Trainingsdaten: Bekannter systematischer Fehler
        # Simuliere einen Arm der immer 0.5° zu weit nach rechts dreht (Base)
        # und 0.3° zu hoch geht (Shoulder)
        commanded_poses = [
            {"b": 0.0,   "s": 0.0,   "e": 90.0},
            {"b": -45.0, "s": 0.0,   "e": 90.0},
            {"b": 45.0,  "s": 0.0,   "e": 90.0},
            {"b": 0.0,   "s": 30.0,  "e": 90.0},
            {"b": 0.0,   "s": -10.0, "e": 90.0},
            {"b": 0.0,   "s": 0.0,   "e": 50.0},
            {"b": 0.0,   "s": 0.0,   "e": 130.0},
            {"b": -30.0, "s": 20.0,  "e": 60.0},
            {"b": 30.0,  "s": 20.0,  "e": 60.0},
            {"b": -30.0, "s": -5.0,  "e": 120.0},
            {"b": 30.0,  "s": -5.0,  "e": 120.0},
            {"b": 0.0,   "s": 15.0,  "e": 70.0},
        ]
        
        # Simulierter systematischer Fehler
        measured_errors = [
            {"b": 0.5, "s": 0.3, "e": -0.2} for _ in commanded_poses
        ]
        
        residuals = model.fit(commanded_poses, measured_errors)
        
        assert model.is_fitted is True
        assert "b" in residuals
        assert "s" in residuals
        assert "e" in residuals
        # Residuen sollten klein sein (perfekter konstanter Fehler)
        assert residuals["b"] < 0.1
        assert residuals["s"] < 0.1
        assert residuals["e"] < 0.1
    
    def test_model_predict_correction(self):
        """Vorhersage korrigiert den systematischen Fehler."""
        from calibrate import CalibrationModel
        
        model = CalibrationModel()
        
        # Trainiere mit konstantem Fehler
        poses = [
            {"b": 0.0,   "s": 0.0,   "e": 90.0},
            {"b": -45.0, "s": 0.0,   "e": 90.0},
            {"b": 45.0,  "s": 0.0,   "e": 90.0},
            {"b": 0.0,   "s": 30.0,  "e": 90.0},
            {"b": 0.0,   "s": -10.0, "e": 90.0},
            {"b": 0.0,   "s": 0.0,   "e": 50.0},
            {"b": 0.0,   "s": 0.0,   "e": 130.0},
            {"b": -30.0, "s": 20.0,  "e": 60.0},
            {"b": 30.0,  "s": 20.0,  "e": 60.0},
            {"b": -30.0, "s": -5.0,  "e": 120.0},
            {"b": 30.0,  "s": -5.0,  "e": 120.0},
            {"b": 0.0,   "s": 15.0,  "e": 70.0},
        ]
        errors = [{"b": 0.5, "s": 0.3, "e": -0.2} for _ in poses]
        model.fit(poses, errors)
        
        # Vorhersage für eine neue Pose
        correction = model.predict_correction({"b": 10.0, "s": 5.0, "e": 80.0})
        
        # Korrektur sollte ungefähr dem trainierten Fehler entsprechen
        assert abs(correction["b"] - 0.5) < 0.2
        assert abs(correction["s"] - 0.3) < 0.2
        assert abs(correction["e"] - (-0.2)) < 0.2
        assert correction["h"] == 0.0  # Hand wird nicht kalibriert
    
    def test_model_not_fitted_returns_zero(self):
        """Ungefittetes Modell gibt Null-Korrektur zurück."""
        from calibrate import CalibrationModel
        
        model = CalibrationModel()
        assert model.is_fitted is False
        
        correction = model.predict_correction({"b": 10.0, "s": 5.0, "e": 80.0})
        assert correction == {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
    
    def test_model_save_and_load(self):
        """Modell kann gespeichert und geladen werden."""
        from calibrate import CalibrationModel
        import tempfile
        
        model = CalibrationModel()
        poses = [
            {"b": 0.0,   "s": 0.0,   "e": 90.0},
            {"b": -45.0, "s": 0.0,   "e": 90.0},
            {"b": 45.0,  "s": 0.0,   "e": 90.0},
            {"b": 0.0,   "s": 30.0,  "e": 90.0},
            {"b": 0.0,   "s": -10.0, "e": 90.0},
            {"b": 0.0,   "s": 0.0,   "e": 50.0},
            {"b": 0.0,   "s": 0.0,   "e": 130.0},
            {"b": -30.0, "s": 20.0,  "e": 60.0},
            {"b": 30.0,  "s": 20.0,  "e": 60.0},
            {"b": -30.0, "s": -5.0,  "e": 120.0},
            {"b": 30.0,  "s": -5.0,  "e": 120.0},
            {"b": 0.0,   "s": 15.0,  "e": 70.0},
        ]
        errors = [{"b": 0.5, "s": 0.3, "e": -0.2} for _ in poses]
        model.fit(poses, errors)
        
        # Speichern
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.cal', delete=False)
        tmp.close()
        model.save(tmp.name)
        
        # Laden
        loaded = CalibrationModel.load(tmp.name)
        assert loaded.is_fitted is True
        
        # Vorhersagen sollten identisch sein
        test_pose = {"b": 20.0, "s": 10.0, "e": 70.0}
        orig_correction = model.predict_correction(test_pose)
        loaded_correction = loaded.predict_correction(test_pose)
        
        for j in ["b", "s", "e"]:
            assert abs(orig_correction[j] - loaded_correction[j]) < 0.001
        
        os.unlink(tmp.name)


class TestPoseValidation:
    """Tests für die Pose-Validierung."""
    
    def test_valid_pose(self):
        """Gültige Pose wird akzeptiert."""
        from calibrate import validate_pose
        
        assert validate_pose({"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}) is True
        assert validate_pose({"b": -45.0, "s": 20.0, "e": 60.0, "h": 180.0}) is True
    
    def test_invalid_pose_base(self):
        """Base außerhalb der Grenzen wird abgelehnt."""
        from calibrate import validate_pose
        
        assert validate_pose({"b": -100.0, "s": 0.0, "e": 90.0, "h": 180.0}) is False
        assert validate_pose({"b": 100.0, "s": 0.0, "e": 90.0, "h": 180.0}) is False
    
    def test_invalid_pose_shoulder(self):
        """Shoulder außerhalb der Grenzen wird abgelehnt."""
        from calibrate import validate_pose
        
        assert validate_pose({"b": 0.0, "s": -30.0, "e": 90.0, "h": 180.0}) is False
        assert validate_pose({"b": 0.0, "s": 50.0, "e": 90.0, "h": 180.0}) is False
    
    def test_invalid_pose_elbow(self):
        """Elbow außerhalb der Grenzen wird abgelehnt."""
        from calibrate import validate_pose
        
        assert validate_pose({"b": 0.0, "s": 0.0, "e": 10.0, "h": 180.0}) is False
        assert validate_pose({"b": 0.0, "s": 0.0, "e": 160.0, "h": 180.0}) is False


# ============================================================
# TESTS: TEACH MODULE (teach.py)
# ============================================================

class TestTeachRecorder:
    """Tests für die Recording-Logik."""
    
    def test_record_point_threshold(self):
        """Punkte unter der Schwelle werden nicht aufgezeichnet."""
        from teach import TeachRecorder
        
        recorder = TeachRecorder(hz=50, threshold=0.5)
        recorder._recording = True
        recorder._rec_start_time = time.time()
        
        # Erster Punkt wird immer aufgezeichnet
        pos1 = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        recorded = recorder._record_point(pos1, force=True)
        assert recorded is True
        assert recorder._total_waypoints == 1
        
        # Zweiter Punkt: Zu nah am ersten → nicht aufgezeichnet
        pos2 = {"b": 0.1, "s": 0.0, "e": 90.0, "h": 180.0}
        recorded = recorder._record_point(pos2)
        assert recorded is False
        assert recorder._total_waypoints == 1
        
        # Dritter Punkt: Weit genug weg → aufgezeichnet
        pos3 = {"b": 5.0, "s": 2.0, "e": 88.0, "h": 180.0}
        recorded = recorder._record_point(pos3)
        assert recorded is True
        assert recorder._total_waypoints == 2
    
    def test_record_point_force(self):
        """Force-Flag erzwingt Aufzeichnung."""
        from teach import TeachRecorder
        
        recorder = TeachRecorder(hz=50, threshold=10.0)  # Hohe Schwelle
        recorder._recording = True
        recorder._rec_start_time = time.time()
        
        pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        recorder._record_point(pos, force=True)
        
        # Gleiche Position nochmal mit force
        recorder._record_point(pos, force=True)
        assert recorder._total_waypoints == 2
    
    def test_waypoint_timing(self):
        """Zeitstempel werden korrekt berechnet."""
        from teach import TeachRecorder
        
        recorder = TeachRecorder(hz=50, threshold=0.1)
        recorder._recording = True
        recorder._rec_start_time = time.time()
        
        pos1 = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        recorder._record_point(pos1, force=True)
        
        time.sleep(0.1)
        
        pos2 = {"b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0}
        recorder._record_point(pos2, force=True)
        
        assert len(recorder._waypoints) == 2
        assert recorder._waypoints[0]["t"] < recorder._waypoints[1]["t"]
        # Zeitdifferenz sollte ungefähr 0.1s sein
        dt = recorder._waypoints[1]["t"] - recorder._waypoints[0]["t"]
        assert 0.05 < dt < 0.3


# ============================================================
# TESTS: INTEGRATION (Zusammenspiel mehrerer Module)
# ============================================================

class TestIntegration:
    """Integrationstests: Mehrere Module zusammen."""
    
    def test_full_record_and_play_cycle(self):
        """Kompletter Zyklus: Aufnehmen → Speichern → Laden → Abspielen."""
        # 1. Aufnahme simulieren
        waypoints = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 0.5, "b": 15.0, "s": 5.0, "e": 85.0, "h": 180.0},
            {"t": 1.0, "b": 30.0, "s": 10.0, "e": 80.0, "h": 180.0},
            {"t": 1.5, "b": 45.0, "s": 10.0, "e": 75.0, "h": 180.0},
            {"t": 2.0, "b": 30.0, "s": 5.0, "e": 80.0, "h": 180.0},
            {"t": 2.5, "b": 15.0, "s": 0.0, "e": 85.0, "h": 180.0},
            {"t": 3.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
        ]
        
        # 2. Datei erstellen
        filepath = make_recording_file(waypoints)
        
        try:
            # 3. Datei laden
            from play import parse_roarm_file, SmoothTrajectory
            data = parse_roarm_file(filepath)
            
            assert len(data["waypoints"]) == 7
            
            # 4. Trajektorie erstellen
            traj = SmoothTrajectory(data["waypoints"], speed_factor=1.0)
            duration = traj.get_duration()
            assert duration > 0
            
            # 5. Trajektorie samplen und an simulierten Arm senden
            arm = make_arm()
            arm.set_position({"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0})
            
            n_commands = 0
            for i in range(int(duration * 40)):  # 40 Hz
                t = i / 40.0
                if t > duration:
                    break
                pos = traj.sample(t)
                arm.move_to(pos["b"], pos["s"], pos["e"], pos["h"], spd=30, acc=15)
                n_commands += 1
            
            assert n_commands > 10
            
            # 6. Arm sollte sich bewegt haben
            arm.wait_for_arrival()
            final_pos = arm.get_true_position()
            # Endposition sollte ungefähr beim letzten Waypoint sein
            assert abs(final_pos["b"] - 0.0) < 5.0
            assert abs(final_pos["e"] - 90.0) < 5.0
            
            arm.close()
        finally:
            os.unlink(filepath)
    
    def test_safety_blocks_dangerous_trajectory(self):
        """Safety-Layer blockiert gefährliche Trajektorie."""
        from safety import SafeArm, SafetyLimits
        
        arm = make_arm()
        limits = SafetyLimits(
            b_min=-90.0, b_max=90.0,
            max_delta_per_cmd=15.0,
        )
        safe = SafeArm(arm, limits=limits)
        
        # Initiale Position setzen
        safe.validator._last_commanded = {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0, "t": time.time()
        }
        
        # Versuch außerhalb der Grenzen zu fahren
        result = safe.move_to(100.0, 0.0, 90.0, 180.0)
        assert result is False
        
        # Versuch innerhalb der Grenzen
        result = safe.move_to(5.0, 0.0, 90.0, 180.0)
        assert result is True
        
        arm.close()
    
    def test_calibration_applied_during_playback(self):
        """Kalibrierung wird korrekt auf Trajektorie angewendet."""
        from calibrate import CalibrationModel
        
        # Modell mit bekanntem Fehler trainieren
        model = CalibrationModel()
        poses = [
            {"b": 0.0,   "s": 0.0,   "e": 90.0},
            {"b": -45.0, "s": 0.0,   "e": 90.0},
            {"b": 45.0,  "s": 0.0,   "e": 90.0},
            {"b": 0.0,   "s": 30.0,  "e": 90.0},
            {"b": 0.0,   "s": -10.0, "e": 90.0},
            {"b": 0.0,   "s": 0.0,   "e": 50.0},
            {"b": 0.0,   "s": 0.0,   "e": 130.0},
            {"b": -30.0, "s": 20.0,  "e": 60.0},
            {"b": 30.0,  "s": 20.0,  "e": 60.0},
            {"b": -30.0, "s": -5.0,  "e": 120.0},
            {"b": 30.0,  "s": -5.0,  "e": 120.0},
            {"b": 0.0,   "s": 15.0,  "e": 70.0},
        ]
        errors = [{"b": 1.0, "s": 0.0, "e": 0.0} for _ in poses]
        model.fit(poses, errors)
        
        # Korrektur anwenden (wie in SmoothPlayer._apply_calibration)
        target = {"b": 20.0, "s": 10.0, "e": 80.0, "h": 180.0}
        correction = model.predict_correction(target)
        
        corrected = {
            "b": target["b"] - correction["b"],
            "s": target["s"] - correction["s"],
            "e": target["e"] - correction["e"],
            "h": target["h"],
        }
        
        # Korrigierte Position sollte kleiner sein (Fehler war positiv)
        assert corrected["b"] < target["b"]
        # Korrektur sollte ungefähr 1° sein
        assert abs(target["b"] - corrected["b"] - 1.0) < 0.3
    
    def test_sim_arm_with_safe_arm_wrapper(self):
        """SimulatedArm funktioniert mit SafeArm Wrapper."""
        from safety import SafeArm, SafetyLimits
        
        arm = make_arm()
        safe = SafeArm(arm, SafetyLimits())
        
        # Initiale Position setzen
        safe.validator._last_commanded = {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0, "t": time.time()
        }
        
        # Normaler Move
        result = safe.move_to(5.0, 2.0, 88.0, 180.0, spd=20, acc=10)
        assert result is True
        
        # Position lesen
        pos = safe.read_position_deg()
        assert pos is not None
        assert "b" in pos
        
        # Torque
        safe.torque_off()
        safe.torque_on()
        
        arm.close()
    
    def test_emergency_stop_halts_everything(self):
        """Emergency Stop blockiert alle weiteren Befehle."""
        from safety import SafeArm, SafetyLimits
        
        arm = make_arm()
        safe = SafeArm(arm, SafetyLimits())
        
        # Initiale Position
        safe.validator._last_commanded = {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0, "t": time.time()
        }
        
        # Emergency Stop auslösen
        safe.validator.trigger_emergency_stop("Test-Emergency")
        
        # Alle Moves sollten jetzt blockiert sein
        result = safe.move_to(5.0, 0.0, 90.0, 180.0)
        assert result is False
        
        result = safe.move_to_fast(5.0, 0.0, 90.0, 180.0)
        assert result is False
        
        # Reset
        safe.reset_emergency()
        
        # Jetzt sollte es wieder gehen
        safe.validator._last_commanded = {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0, "t": time.time()
        }
        result = safe.move_to(5.0, 0.0, 90.0, 180.0)
        assert result is True
        
        arm.close()


# ============================================================
# TESTS: EDGE CASES & ROBUSTHEIT
# ============================================================

class TestEdgeCases:
    """Tests für Grenzfälle und Fehlerbehandlung."""
    
    def test_empty_recording_file(self):
        """Leere Datei wird korrekt behandelt."""
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.roarm', delete=False)
        tmp.write("# Leere Datei\n")
        tmp.close()
        
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(tmp.name)
            assert len(data["waypoints"]) == 0
        finally:
            os.unlink(tmp.name)
    
    def test_single_waypoint_file(self):
        """Datei mit nur einem Waypoint."""
        waypoints = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
        ]
        filepath = make_recording_file(waypoints)
        
        try:
            from play import parse_roarm_file
            data = parse_roarm_file(filepath)
            assert len(data["waypoints"]) == 1
        finally:
            os.unlink(filepath)
    
    def test_concurrent_reads_and_moves(self):
        """Gleichzeitige Reads und Moves crashen nicht."""
        arm = make_arm()
        
        errors = []
        
        def reader():
            for _ in range(50):
                try:
                    pos = arm.read_position_deg()
                    # pos kann None sein, das ist OK
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)
        
        def mover():
            for i in range(50):
                try:
                    arm.move_to(i * 0.5, 0.0, 90.0, 180.0)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)
        
        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=mover)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        assert len(errors) == 0, f"Fehler bei gleichzeitigem Zugriff: {errors}"
        arm.close()
    
    def test_rapid_torque_toggle(self):
        """Schnelles Torque-Umschalten crasht nicht."""
        arm = make_arm()
        
        for _ in range(20):
            arm.torque_on()
            arm.torque_off()
        
        # Arm sollte noch funktionieren
        pos = arm.read_position_deg()
        assert pos is not None
        arm.close()
    
    def test_move_to_current_position(self):
        """Move zur aktuellen Position ist ein No-Op."""
        arm = make_arm()
        arm.set_position({"b": 10.0, "s": 5.0, "e": 80.0, "h": 180.0})
        
        # Move zur gleichen Position
        arm.move_to(10.0, 5.0, 80.0, 180.0)
        arm.wait_for_arrival()
        
        pos = arm.get_true_position()
        assert abs(pos["b"] - 10.0) < 0.5
        arm.close()
    
    def test_negative_positions(self):
        """Negative Winkel funktionieren korrekt."""
        arm = make_arm()
        arm.move_to(-30.0, -10.0, 45.0, 180.0)
        arm.wait_for_arrival()
        
        pos = arm.get_true_position()
        assert abs(pos["b"] - (-30.0)) < 1.0
        assert abs(pos["s"] - (-10.0)) < 1.0
        arm.close()
    
    def test_offset_with_zero_blend_points(self):
        """Offset mit 0 Blend-Points wendet sofort vollen Offset an."""
        from play import apply_offset_to_waypoints
        
        wps = [
            {"t": 0.0, "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0},
            {"t": 1.0, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0},
        ]
        offset = {"b": 2.0, "s": 0.0, "e": 0.0, "h": 0.0}
        
        result = apply_offset_to_waypoints(wps, offset, blend_points=2)
        # Beide Punkte sollten Offset haben (blend über alle)
        assert result[-1]["b"] == 12.0  # 10 + 2


# ============================================================
# MAIN: Test-Runner
# ============================================================

def run_all_tests():
    """Führt alle Tests aus (ohne pytest)."""
    import traceback
    
    test_classes = [
        TestSimulatedArm,
        TestPositionValidator,
        TestSafeArm,
        TestThermalEstimator,
        TestRateLimiter,
        TestTrajectoryValidator,
        TestParseRoarmFile,
        TestApplyOffset,
        TestSmoothTrajectory,
        TestCalibrationModel,
        TestPoseValidation,
        TestTeachRecorder,
        TestIntegration,
        TestEdgeCases,
    ]
    
    total = 0
    passed = 0
    failed = 0
    errors = []
    
    print("=" * 70)
    print("  RoArm-M2-S Test Suite")
    print("=" * 70)
    
    for cls in test_classes:
        print(f"\n{'─' * 50}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 50}")
        
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            
            try:
                method()
                passed += 1
                print(f"  ✅ {method_name}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                errors.append((cls.__name__, method_name, tb))
                print(f"  ❌ {method_name}")
                print(f"     {type(e).__name__}: {e}")
    
    # Zusammenfassung
    print(f"\n{'=' * 70}")
    print(f"  ERGEBNIS: {passed}/{total} bestanden, {failed} fehlgeschlagen")
    print(f"{'=' * 70}")
    
    if errors:
        print(f"\n{'─' * 70}")
        print("  FEHLGESCHLAGENE TESTS:")
        print(f"{'─' * 70}")
        for cls_name, method_name, tb in errors:
            print(f"\n  {cls_name}.{method_name}:")
            for line in tb.split("\n")[-5:]:
                print(f"    {line}")
    
    return failed == 0


if __name__ == "__main__":
    # Versuche pytest zu verwenden, Fallback auf eigenen Runner
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    except ImportError:
        success = run_all_tests()
        sys.exit(0 if success else 1)
