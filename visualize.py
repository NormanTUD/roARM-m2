#!/usr/bin/env python3
"""visualize.py - 3D-Visualisierung des RoArm-M2-S

ARCHITEKTUR:
  Matplotlib MUSS im Main-Thread laufen (Tk-Requirement).
  Daher: Separater Prozess für die Visualisierung.

KINEMATIK des RoArm-M2-S:
  - Base (b): 360° Rotation um vertikale Z-Achse (wir nutzen ±180°)
  - Shoulder (s): 180° Range. s=0° = Oberarm horizontal nach vorne
                  s>0 = Oberarm nach oben, s<0 = nach unten
  - Elbow (e): 135°/270° Range. Winkel ZWISCHEN Ober- und Unterarm.
               e=180° = gestreckt (Unterarm = Verlängerung Oberarm)
               e=90° = Unterarm steht 90° zum Oberarm (nach oben/innen)
               e=0° = Unterarm klappt komplett zurück
  - Gripper: direkt am Elbow-Segment befestigt (kein separates Wrist-Gelenk)

PROPORTIONEN (Waveshare RoArm-M2-S Specs):
  - Basis-Höhe: ~75mm (Drehplattform + Shoulder-Gelenk)
  - Oberarm (Shoulder → Elbow): ~206mm
  - Unterarm (Elbow → Gripper-Spitze): ~286mm (206mm Unterarm + 80mm Gripper)
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from typing import Optional
import threading
import multiprocessing
import queue
import time
import os


# ============================================================
# KINEMATIK-KONSTANTEN
# ============================================================

BASE_HEIGHT = 75.0       # Höhe Basis bis Shoulder-Gelenk
UPPER_ARM = 206.0        # Oberarm: Shoulder → Elbow
FOREARM = 206.0          # Unterarm: Elbow → Gripper-Ansatz
GRIPPER_LENGTH = 80.0    # Gripper-Länge (Teil des Unterarm-Segments)


# ============================================================
# FORWARD KINEMATIK (ohne Wrist)
# ============================================================

def forward_kinematics(b_deg: float, s_deg: float, e_deg: float) -> dict:
    """
    Kinematik für RoArm-M2-S (ohne Wrist-Gelenk).
    
    Gelenke:
    - b: Base-Rotation um Z-Achse. b=0 → +X (nach vorne)
    - s: Shoulder. s=0 → Oberarm HORIZONTAL nach vorne.
         s>0 → nach oben, s<0 → nach unten.
    - e: Elbow. Innenwinkel zwischen Ober- und Unterarm.
         e=180° → gestreckt
         e=90° → Unterarm steht 90° zum Oberarm
    """
    b_rad = math.radians(b_deg)
    s_rad = math.radians(90.0 - s_deg)

    base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, BASE_HEIGHT])

    # Oberarm
    elbow_local_x = UPPER_ARM * math.cos(s_rad)
    elbow_local_z = BASE_HEIGHT + UPPER_ARM * math.sin(s_rad)

    # Unterarm: absoluter Winkel
    forearm_abs_angle = s_rad - math.radians(e_deg)

    # Gripper-Spitze (Unterarm + Gripper als ein Segment)
    total_forearm = FOREARM + GRIPPER_LENGTH
    gripper_local_x = elbow_local_x + total_forearm * math.cos(forearm_abs_angle)
    gripper_local_z = elbow_local_z + total_forearm * math.sin(forearm_abs_angle)

    # Base-Rotation um Z-Achse
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


# ============================================================
# VISUALIZER PROCESS
# ============================================================

def _visualizer_process_main(pose_queue: multiprocessing.Queue,
                              control_queue: multiprocessing.Queue,
                              update_interval: float = 0.05):
    """
    Hauptfunktion des Visualizer-Prozesses.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    current_pose = {"b": 0.0, "s": 0.0, "e": 90.0}
    target_pose = None
    trail = []
    max_trail = 200

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("RoArm-M2-S 3D Visualisierung")
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

    JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Gripper']

    def configure_axes():
        limit = 500.0
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-50, limit])
        ax.set_xlabel("X [mm] (vorne)", fontsize=9)
        ax.set_ylabel("Y [mm] (links)", fontsize=9)
        ax.set_zlabel("Z [mm] (oben)", fontsize=9)
        ax.set_title("RoArm-M2-S", fontsize=14, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=-60)
        ax.set_proj_type('persp', focal_length=0.5)

    def draw_arm():
        ax.cla()
        configure_axes()

        positions = forward_kinematics(
            current_pose["b"], current_pose["s"], current_pose["e"]
        )

        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["gripper"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # === ARM SEGMENTE ===
        # Basis (Drehsäule)
        ax.plot([0, 0], [0, 0], [0, BASE_HEIGHT],
                color='#424242', linewidth=10, solid_capstyle='round', alpha=0.8)
        
        # Oberarm (Shoulder → Elbow) - blau
        ax.plot([xs[1], xs[2]], [ys[1], ys[2]], [zs[1], zs[2]],
                color='#1565C0', linewidth=7, solid_capstyle='round', label='Oberarm')
        
        # Unterarm + Gripper (Elbow → Gripper) - grün/orange
        ax.plot([xs[2], xs[3]], [ys[2], ys[3]], [zs[2], zs[3]],
                color='#2E7D32', linewidth=6, solid_capstyle='round', label='Unterarm + Gripper')

        # === GELENKE mit Beschriftung ===
        joint_colors = ['#212121', '#D32F2F', '#1565C0', '#E65100']
        joint_sizes = [100, 80, 70, 60]
        
        for i, (x, y, z) in enumerate(pts):
            ax.scatter([x], [y], [z], c=joint_colors[i], s=joint_sizes[i],
                      zorder=5, depthshade=False, edgecolors='white', linewidths=0.5)
            offset_z = 20 if i < 3 else -25
            ax.text(x, y, z + offset_z, JOINT_NAMES[i],
                    fontsize=8, ha='center', va='bottom' if offset_z > 0 else 'top',
                    color=joint_colors[i], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             alpha=0.7, edgecolor=joint_colors[i], linewidth=0.5))

        # === BODEN-PROJEKTION (Schatten) ===
        ax.plot(xs, ys, [0]*len(xs), color='gray', linewidth=1,
                linestyle=':', alpha=0.3)
        ax.plot([xs[-1], xs[-1]], [ys[-1], ys[-1]], [0, zs[-1]],
                color='gray', linewidth=0.5, linestyle=':', alpha=0.2)

        # === ENDEFFEKTOR-TRAIL ===
        if trail:
            trail_x = [p[0] for p in trail]
            trail_y = [p[1] for p in trail]
            trail_z = [p[2] for p in trail]
            ax.plot(trail_x, trail_y, trail_z,
                    color='red', linewidth=1.5, alpha=0.5)

        # === TARGET ===
        if target_pose:
            tp = forward_kinematics(
                target_pose["b"], target_pose["s"], target_pose["e"]
            )["gripper"]
            ax.scatter([tp[0]], [tp[1]], [tp[2]],
                      c='red', s=120, marker='x', linewidths=3, zorder=6)

        # === KOORDINATENACHSEN am Ursprung ===
        axis_len = 60.0
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.15, 
                  alpha=0.6, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.15, 
                  alpha=0.6, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.15, 
                  alpha=0.6, linewidth=1.5)
        ax.text(axis_len + 10, 0, 0, "X", color='red', fontsize=8)
        ax.text(0, axis_len + 10, 0, "Y", color='green', fontsize=8)
        ax.text(0, 0, axis_len + 10, "Z", color='blue', fontsize=8)

        # === ARBEITSRAUM-KREIS (Boden) ===
        theta = np.linspace(0, 2*np.pi, 80)
        reach = UPPER_ARM + FOREARM + GRIPPER_LENGTH
        ax.plot(reach * np.cos(theta), reach * np.sin(theta),
                np.zeros(80), color='gray', linewidth=0.5, alpha=0.15)
        half_reach = UPPER_ARM
        ax.plot(half_reach * np.cos(theta), half_reach * np.sin(theta),
                np.zeros(80), color='gray', linewidth=0.3, alpha=0.1, linestyle='--')

        # === BODEN-PLATTE ===
        plate_size = 80
        plate_x = [-plate_size, plate_size, plate_size, -plate_size, -plate_size]
        plate_y = [-plate_size, -plate_size, plate_size, plate_size, -plate_size]
        ax.plot(plate_x, plate_y, [0]*5, color='#616161', linewidth=1.5, alpha=0.4)

        # === INFO-TEXT ===
        info_text = (
            f"b={current_pose['b']:+6.1f}\u00b0  s={current_pose['s']:+6.1f}\u00b0  "
            f"e={current_pose['e']:+6.1f}\u00b0\n"
            f"Gripper: ({positions['gripper'][0]:.0f}, "
            f"{positions['gripper'][1]:.0f}, {positions['gripper'][2]:.0f}) mm"
        )
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                  fontsize=9, fontfamily='monospace',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # Initial zeichnen
    draw_arm()
    fig.canvas.draw_idle()
    plt.pause(0.01)

    # === MAIN LOOP ===
    running = True
    dirty = False

    while running:
        try:
            while True:
                cmd = control_queue.get_nowait()
                if cmd == "stop":
                    running = False
                    break
                elif cmd == "clear_trail":
                    trail.clear()
                    dirty = True
        except queue.Empty:
            pass

        if not running:
            break

        try:
            while True:
                pose_data = pose_queue.get_nowait()
                current_pose["b"] = pose_data["b"]
                current_pose["s"] = pose_data["s"]
                current_pose["e"] = pose_data["e"]
                target_pose = pose_data.get("target", None)

                positions = forward_kinematics(
                    current_pose["b"], current_pose["s"], current_pose["e"]
                )
                trail.append(positions["gripper"].copy())
                if len(trail) > max_trail:
                    trail.pop(0)

                dirty = True
        except queue.Empty:
            pass

        if dirty:
            draw_arm()
            fig.canvas.draw_idle()
            dirty = False

        if not plt.fignum_exists(fig.number):
            break

        try:
            plt.pause(update_interval)
        except Exception:
            break

    try:
        plt.close(fig)
    except Exception:
        pass


# ============================================================
# ROBOT VISUALIZER (API)
# ============================================================

class RobotVisualizer:
    """
    3D-Visualisierung als separater Prozess.
    """

    def __init__(self, live: bool = False, update_interval: float = 0.05):
        self._live = live
        self._update_interval = update_interval
        self._pose_queue = None
        self._control_queue = None
        self._process = None
        self._running = False

    def start(self):
        if self._running:
            return

        self._pose_queue = multiprocessing.Queue(maxsize=100)
        self._control_queue = multiprocessing.Queue(maxsize=10)

        self._process = multiprocessing.Process(
            target=_visualizer_process_main,
            args=(self._pose_queue, self._control_queue, self._update_interval),
            daemon=True,
        )
        self._process.start()
        self._running = True

    def stop(self):
        if not self._running:
            return

        self._running = False

        try:
            self._control_queue.put_nowait("stop")
        except Exception:
            pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()

        self._process = None

    def update_pose(self, b: float, s: float, e: float, h: float,
                    target: dict = None):
        """Sendet eine neue Pose an den Visualizer (thread-safe)."""
        if not self._running:
            return

        pose_data = {"b": b, "s": s, "e": e}
        if target:
            pose_data["target"] = target

        try:
            if self._pose_queue.full():
                try:
                    self._pose_queue.get_nowait()
                except queue.Empty:
                    pass
            self._pose_queue.put_nowait(pose_data)
        except Exception:
            pass

    def clear_trail(self):
        if self._running:
            try:
                self._control_queue.put_nowait("clear_trail")
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        if self._process and not self._process.is_alive():
            self._running = False
        return self._running

    def show_pose(self, b: float = 0.0, s: float = 0.0, e: float = 90.0):
        """Zeigt eine einzelne Pose (blockierend, Main-Thread)."""
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title("RoArm-M2-S 3D Visualisierung")
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

        pose = {"b": b, "s": s, "e": e}
        self._draw_arm_static(ax, pose)

        fig.canvas.draw_idle()
        plt.ioff()
        plt.show()

    def show_trajectory(self, waypoints: list, fps: int = 30):
        """Animiert eine Trajektorie (blockierend, Main-Thread)."""
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title("RoArm-M2-S 3D Visualisierung")
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

        trail = []
        for wp in waypoints:
            pose = {"b": wp["b"], "s": wp["s"], "e": wp["e"]}
            positions = forward_kinematics(pose["b"], pose["s"], pose["e"])
            trail.append(positions["gripper"])
            if len(trail) > 200:
                trail.pop(0)

            self._draw_arm_static(ax, pose, trail=trail)
            fig.canvas.draw_idle()
            plt.pause(1.0 / fps)

        plt.ioff()
        plt.show()

    def _draw_arm_static(self, ax, pose, trail=None, target=None):
        """Zeichnet den Arm (für statische/blockierende Methoden)."""
        ax.cla()

        JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Gripper']
        
        limit = 500.0
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-100, limit])
        ax.set_xlabel("X [mm] (vorne)", fontsize=9)
        ax.set_ylabel("Y [mm] (links)", fontsize=9)
        ax.set_zlabel("Z [mm] (oben)", fontsize=9)
        ax.set_title("RoArm-M2-S", fontsize=14, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])

        positions = forward_kinematics(pose["b"], pose["s"], pose["e"])
        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["gripper"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # Basis
        ax.plot([0, 0], [0, 0], [0, BASE_HEIGHT],
                color='#424242', linewidth=10, solid_capstyle='round', alpha=0.8)
        # Oberarm
        ax.plot([xs[1], xs[2]], [ys[1], ys[2]], [zs[1], zs[2]],
                color='#1565C0', linewidth=7, solid_capstyle='round')
        # Unterarm + Gripper
        ax.plot([xs[2], xs[3]], [ys[2], ys[3]], [zs[2], zs[3]],
                color='#2E7D32', linewidth=6, solid_capstyle='round')

        # Gelenke + Labels
        joint_colors = ['#212121', '#D32F2F', '#1565C0', '#E65100']
        joint_sizes = [100, 80, 70, 60]
        for i, (x, y, z) in enumerate(pts):
            ax.scatter([x], [y], [z], c=joint_colors[i], s=joint_sizes[i],
                      zorder=5, depthshade=False, edgecolors='white', linewidths=0.5)
            offset_z = 20 if i < 3 else -25
            ax.text(x, y, z + offset_z, JOINT_NAMES[i],
                    fontsize=8, ha='center', va='bottom' if offset_z > 0 else 'top',
                    color=joint_colors[i], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             alpha=0.7, edgecolor=joint_colors[i], linewidth=0.5))

        # Trail
        if trail:
            ax.plot([p[0] for p in trail], [p[1] for p in trail],
                    [p[2] for p in trail], color='red', linewidth=1.5, alpha=0.5)

        # Koordinatenachsen
        axis_len = 60.0
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.15, alpha=0.6)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.15, alpha=0.6)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.15, alpha=0.6)

        # Arbeitsraum
        theta = np.linspace(0, 2*np.pi, 80)
        reach = UPPER_ARM + FOREARM + GRIPPER_LENGTH
        ax.plot(reach * np.cos(theta), reach * np.sin(theta),
                np.zeros(80), color='gray', linewidth=0.5, alpha=0.15)

        # Info
        info_text = (
            f"b={pose['b']:+6.1f}\u00b0  s={pose['s']:+6.1f}\u00b0  "
            f"e={pose['e']:+6.1f}\u00b0\n"
            f"Gripper: ({positions['gripper'][0]:.0f}, "
            f"{positions['gripper'][1]:.0f}, {positions['gripper'][2]:.0f}) mm"
        )
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))


# ============================================================
# INTEGRATION: VisualizingArm Wrapper
# ============================================================

class VisualizingArm:
    """
    Wrapper um RoArmConnection der jede Bewegung
    automatisch in der 3D-Visualisierung anzeigt.
    """

    def __init__(self, arm, show_target: bool = True, trail: bool = True):
        self._arm = arm
        self._show_target = show_target
        self._viz = RobotVisualizer(live=True, update_interval=0.05)
        self._viz.start()
        time.sleep(0.5)

        try:
            pos = self._arm.read_position_deg()
            if pos:
                self._viz.update_pose(pos["b"], pos["s"], pos["e"])
        except Exception:
            pass

    def move_to(self, b_deg: float, s_deg: float, e_deg: float,
                spd: int = 20, acc: int = 10):
        target = {"b": b_deg, "s": s_deg, "e": e_deg}
        self._arm.move_to(b_deg, s_deg, e_deg, spd=spd, acc=acc)

        if self._show_target:
            self._viz.update_pose(b_deg, s_deg, e_deg, target=target)
        else:
            self._viz.update_pose(b_deg, s_deg, e_deg)

    def move_to_fast(self, b_deg: float, s_deg: float, e_deg: float,
                     spd: int = 50, acc: int = 30):
        self._arm.move_to_fast(b_deg, s_deg, e_deg, spd=spd, acc=acc)
        self._viz.update_pose(b_deg, s_deg, e_deg)

    def read_position_deg(self) -> Optional[dict]:
        pos = self._arm.read_position_deg()
        if pos:
            self._viz.update_pose(pos["b"], pos["s"], pos["e"])
        return pos

    def torque_on(self):
        self._arm.torque_on()

    def torque_off(self):
        self._arm.torque_off()

    def gripper_open(self):
        self._arm.gripper_open()

    def gripper_close(self):
        self._arm.gripper_close()

    def wait_until_settled(self, **kwargs):
        result = self._arm.wait_until_settled(**kwargs)
        if result and result.get("pos"):
            pos = result["pos"]
            self._viz.update_pose(pos["b"], pos["s"], pos["e"])
        return result

    def send_cmd(self, cmd: dict, **kwargs):
        return self._arm.send_cmd(cmd, **kwargs)

    def close(self):
        self._viz.stop()
        self._arm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def visualizer(self) -> RobotVisualizer:
        return self._viz
