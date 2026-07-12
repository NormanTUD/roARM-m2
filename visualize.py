#!/usr/bin/env python3
"""visualize.py - 3D-Visualisierung des RoArm-M2-S

ARCHITEKTUR:
  Matplotlib MUSS im Main-Thread laufen (Tk-Requirement).
  Daher: Die Visualisierung übernimmt den Main-Thread mit einer
  Timer-basierten Update-Schleife. Die Arm-Logik (play/teach/calibrate)
  läuft in einem Worker-Thread und schreibt Posen in eine Queue.

  Für den Fall dass die Visualisierung NICHT den Main-Thread haben kann
  (z.B. weil play.py den Main-Thread braucht), gibt es einen Fallback
  mit matplotlib.use('Agg') + separatem Prozess.
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
# KINEMATIK-KONSTANTEN (in mm, anpassen an deinen Arm!)
# ============================================================

BASE_HEIGHT = 55.0
UPPER_ARM = 96.0
FOREARM = 96.0
HAND_LENGTH = 70.0


# ============================================================
# FORWARD KINEMATIK
# ============================================================

def forward_kinematics(b_deg: float, s_deg: float, e_deg: float, h_deg: float) -> dict:
    """Berechnet die 3D-Positionen aller Gelenke aus den Winkeln."""
    b_rad = math.radians(b_deg)
    s_rad = math.radians(s_deg)

    base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, BASE_HEIGHT])

    upper_arm_x = UPPER_ARM * math.cos(s_rad)
    upper_arm_z = UPPER_ARM * math.sin(s_rad)

    elbow_local_x = upper_arm_x
    elbow_local_z = BASE_HEIGHT + upper_arm_z

    forearm_angle = s_rad - math.radians(180.0 - e_deg)

    forearm_x = FOREARM * math.cos(forearm_angle)
    forearm_z = FOREARM * math.sin(forearm_angle)

    wrist_local_x = elbow_local_x + forearm_x
    wrist_local_z = elbow_local_z + forearm_z

    hand_x = HAND_LENGTH * math.cos(forearm_angle)
    hand_z = HAND_LENGTH * math.sin(forearm_angle)

    hand_local_x = wrist_local_x + hand_x
    hand_local_z = wrist_local_z + hand_z

    cos_b = math.cos(b_rad)
    sin_b = math.sin(b_rad)

    def rotate_base(x, y, z):
        rx = x * cos_b - y * sin_b
        ry = x * sin_b + y * cos_b
        return np.array([rx, ry, z])

    return {
        "base": base,
        "shoulder": shoulder,
        "elbow": rotate_base(elbow_local_x, 0.0, elbow_local_z),
        "wrist": rotate_base(wrist_local_x, 0.0, wrist_local_z),
        "hand": rotate_base(hand_local_x, 0.0, hand_local_z),
    }


# ============================================================
# VISUALIZER PROCESS (läuft als separater Prozess!)
# ============================================================

def _visualizer_process_main(pose_queue: multiprocessing.Queue,
                              control_queue: multiprocessing.Queue,
                              update_interval: float = 0.05):
    """
    Hauptfunktion des Visualizer-Prozesses.
    Läuft komplett separat → kein Threading-Problem mit Matplotlib.
    
    Kommunikation über Queues:
    - pose_queue: Empfängt Posen-Dicts {"b":..., "s":..., "e":..., "h":..., "target":...}
    - control_queue: Empfängt Steuerbefehle ("stop", "clear_trail")
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    current_pose = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
    target_pose = None
    trail = []
    max_trail = 200

    # Plot erstellen
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("RoArm-M2-S 3D Visualisierung")
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def configure_axes():
        limit = 300.0
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-50, limit])
        ax.set_xlabel("X [mm]", fontsize=9)
        ax.set_ylabel("Y [mm]", fontsize=9)
        ax.set_zlabel("Z [mm]", fontsize=9)
        ax.set_title("RoArm-M2-S", fontsize=14, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])

    def draw_arm():
        ax.cla()
        configure_axes()

        positions = forward_kinematics(
            current_pose["b"], current_pose["s"],
            current_pose["e"], current_pose["h"]
        )

        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["wrist"], positions["hand"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # Arm
        ax.plot([0, 0], [0, 0], [0, BASE_HEIGHT],
                color='gray', linewidth=8, solid_capstyle='round', alpha=0.7)
        ax.plot([xs[1], xs[2]], [ys[1], ys[2]], [zs[1], zs[2]],
                color='#2196F3', linewidth=6, solid_capstyle='round', label='Oberarm')
        ax.plot([xs[2], xs[3]], [ys[2], ys[3]], [zs[2], zs[3]],
                color='#4CAF50', linewidth=5, solid_capstyle='round', label='Unterarm')
        ax.plot([xs[3], xs[4]], [ys[3], ys[4]], [zs[3], zs[4]],
                color='#FF9800', linewidth=4, solid_capstyle='round', label='Hand')

        # Gelenke
        joint_colors = ['black', 'red', 'blue', 'green', 'orange']
        joint_sizes = [80, 60, 50, 40, 35]
        for i, (x, y, z) in enumerate(pts):
            ax.scatter([x], [y], [z], c=joint_colors[i], s=joint_sizes[i],
                      zorder=5, depthshade=False)

        # Schatten
        ax.plot(xs, ys, [0]*len(xs), color='gray', linewidth=1,
                linestyle='--', alpha=0.3)

        # Trail
        if trail:
            trail_x = [p[0] for p in trail]
            trail_y = [p[1] for p in trail]
            trail_z = [p[2] for p in trail]
            ax.plot(trail_x, trail_y, trail_z,
                    color='red', linewidth=1, alpha=0.4)

        # Target
        if target_pose:
            tp = forward_kinematics(
                target_pose["b"], target_pose["s"],
                target_pose["e"], target_pose["h"]
            )["hand"]
            ax.scatter([tp[0]], [tp[1]], [tp[2]],
                      c='red', s=100, marker='x', linewidths=3, zorder=6)

        # Koordinatenachsen
        axis_len = 50.0
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.1, alpha=0.5)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.1, alpha=0.5)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.1, alpha=0.5)

        # Arbeitsraum
        theta = np.linspace(0, 2*np.pi, 50)
        reach = UPPER_ARM + FOREARM + HAND_LENGTH
        ax.plot(reach * np.cos(theta), reach * np.sin(theta),
                np.zeros(50), color='gray', linewidth=0.5, alpha=0.2)

        # Info-Text
        info_text = (
            f"b={current_pose['b']:+6.1f}\u00b0  s={current_pose['s']:+6.1f}\u00b0  "
            f"e={current_pose['e']:+6.1f}\u00b0  h={current_pose['h']:+6.1f}\u00b0\n"
            f"Endeffektor: ({positions['hand'][0]:.0f}, "
            f"{positions['hand'][1]:.0f}, {positions['hand'][2]:.0f}) mm"
        )
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                  fontsize=9, fontfamily='monospace',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(loc='upper right', fontsize=8)

    # Initial zeichnen
    draw_arm()
    fig.canvas.draw_idle()
    plt.pause(0.01)

    # === MAIN LOOP (im Main-Thread dieses Prozesses!) ===
    running = True
    dirty = False

    while running:
        # Steuerbefehle prüfen
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

        # Neue Posen aus der Queue lesen (alle verfügbaren)
        try:
            while True:
                pose_data = pose_queue.get_nowait()
                current_pose["b"] = pose_data["b"]
                current_pose["s"] = pose_data["s"]
                current_pose["e"] = pose_data["e"]
                current_pose["h"] = pose_data["h"]
                target_pose = pose_data.get("target", None)

                # Trail updaten
                positions = forward_kinematics(
                    current_pose["b"], current_pose["s"],
                    current_pose["e"], current_pose["h"]
                )
                trail.append(positions["hand"].copy())
                if len(trail) > max_trail:
                    trail.pop(0)

                dirty = True
        except queue.Empty:
            pass

        # Nur neu zeichnen wenn sich was geändert hat
        if dirty:
            draw_arm()
            fig.canvas.draw_idle()
            dirty = False

        # Prüfe ob Fenster noch offen
        if not plt.fignum_exists(fig.number):
            break

        # Event-Loop bedienen (WICHTIG!)
        try:
            plt.pause(update_interval)
        except Exception:
            break

    # Aufräumen
    try:
        plt.close(fig)
    except Exception:
        pass


# ============================================================
# ROBOT VISUALIZER (API für play.py/teach.py/calibrate.py)
# ============================================================

class RobotVisualizer:
    """
    3D-Visualisierung als separater Prozess.
    
    Startet einen eigenen Prozess für Matplotlib → kein Threading-Problem!
    Kommunikation über multiprocessing.Queue (thread-safe UND process-safe).
    """

    def __init__(self, live: bool = False, update_interval: float = 0.05):
        self._live = live
        self._update_interval = update_interval
        self._pose_queue = None
        self._control_queue = None
        self._process = None
        self._running = False

    def start(self):
        """Startet den Visualizer-Prozess."""
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
        """Stoppt den Visualizer-Prozess."""
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

        pose_data = {"b": b, "s": s, "e": e, "h": h}
        if target:
            pose_data["target"] = target

        try:
            # Non-blocking put - wenn Queue voll, älteste verwerfen
            if self._pose_queue.full():
                try:
                    self._pose_queue.get_nowait()
                except queue.Empty:
                    pass
            self._pose_queue.put_nowait(pose_data)
        except Exception:
            pass  # Queue-Fehler ignorieren (Prozess evtl. schon beendet)

    def clear_trail(self):
        """Löscht den Endeffektor-Pfad."""
        if self._running:
            try:
                self._control_queue.put_nowait("clear_trail")
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        """Prüft ob der Visualizer noch läuft."""
        if self._process and not self._process.is_alive():
            self._running = False
        return self._running

    # --- Standalone-Methoden (blockierend, für Demos) ---

    def show_pose(self, b: float = 0.0, s: float = 0.0,
                  e: float = 90.0, h: float = 180.0):
        """Zeigt eine einzelne Pose (blockierend, Main-Thread)."""
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title("RoArm-M2-S 3D Visualisierung")
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        pose = {"b": b, "s": s, "e": e, "h": h}
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
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        trail = []
        for wp in waypoints:
            pose = {"b": wp["b"], "s": wp["s"], "e": wp["e"], "h": wp["h"]}
            positions = forward_kinematics(pose["b"], pose["s"], pose["e"], pose["h"])
            trail.append(positions["hand"])
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

        limit = 300.0
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-50, limit])
        ax.set_xlabel("X [mm]", fontsize=9)
        ax.set_ylabel("Y [mm]", fontsize=9)
        ax.set_zlabel("Z [mm]", fontsize=9)
        ax.set_title("RoArm-M2-S", fontsize=14, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])

        positions = forward_kinematics(pose["b"], pose["s"], pose["e"], pose["h"])
        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["wrist"], positions["hand"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        ax.plot([0, 0], [0, 0], [0, BASE_HEIGHT],
                color='gray', linewidth=8, solid_capstyle='round', alpha=0.7)
        ax.plot([xs[1], xs[2]], [ys[1], ys[2]], [zs[1], zs[2]],
                color='#2196F3', linewidth=6, solid_capstyle='round')
        ax.plot([xs[2], xs[3]], [ys[2], ys[3]], [zs[2], zs[3]],
                color='#4CAF50', linewidth=5, solid_capstyle='round')
        ax.plot([xs[3], xs[4]], [ys[3], ys[4]], [zs[3], zs[4]],
                color='#FF9800', linewidth=4, solid_capstyle='round')

        joint_colors = ['black', 'red', 'blue', 'green', 'orange']
        joint_sizes = [80, 60, 50, 40, 35]
        for i, (x, y, z) in enumerate(pts):
            ax.scatter([x], [y], [z], c=joint_colors[i], s=joint_sizes[i],
                      zorder=5, depthshade=False)

        if trail:
            ax.plot([p[0] for p in trail], [p[1] for p in trail],
                    [p[2] for p in trail], color='red', linewidth=1, alpha=0.4)

        info_text = (
            f"b={pose['b']:+6.1f}\u00b0  s={pose['s']:+6.1f}\u00b0  "
            f"e={pose['e']:+6.1f}\u00b0  h={pose['h']:+6.1f}\u00b0"
        )
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


# ============================================================
# INTEGRATION: VisualizingArm Wrapper
# ============================================================

class VisualizingArm:
    """
    Wrapper um RoArmConnection der jede Bewegung
    automatisch in der 3D-Visualisierung anzeigt.
    
    Verwendet einen separaten PROZESS für Matplotlib →
    funktioniert aus jedem Thread heraus!
    """

    def __init__(self, arm, show_target: bool = True, trail: bool = True):
        self._arm = arm
        self._show_target = show_target
        self._viz = RobotVisualizer(live=True, update_interval=0.05)
        self._viz.start()

        # Kurz warten bis Prozess gestartet ist
        time.sleep(0.5)

        # Initiale Position lesen und anzeigen
        try:
            pos = self._arm.read_position_deg()
            if pos:
                self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])
        except Exception:
            pass

    def move_to(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                spd: int = 20, acc: int = 10):
        """Move mit Visualisierung."""
        target = {"b": b_deg, "s": s_deg, "e": e_deg, "h": h_deg}
        self._arm.move_to(b_deg, s_deg, e_deg, h_deg, spd=spd, acc=acc)

        if self._show_target:
            self._viz.update_pose(b_deg, s_deg, e_deg, h_deg, target=target)
        else:
            self._viz.update_pose(b_deg, s_deg, e_deg, h_deg)

    def move_to_fast(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                     spd: int = 50, acc: int = 30):
        """Fast-Move mit Visualisierung."""
        self._arm.move_to_fast(b_deg, s_deg, e_deg, h_deg, spd=spd, acc=acc)
        self._viz.update_pose(b_deg, s_deg, e_deg, h_deg)

    def read_position_deg(self) -> Optional[dict]:
        """Liest Position und aktualisiert Visualisierung."""
        pos = self._arm.read_position_deg()
        if pos:
            self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])
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
            self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])
        return result

    def send_cmd(self, cmd: dict, **kwargs):
        return self._arm.send_cmd(cmd, **kwargs)

    def close(self):
        """Schließt Arm und Visualisierung."""
        self._viz.stop()
        self._arm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def visualizer(self) -> RobotVisualizer:
        """Zugriff auf den Visualizer."""
        return self._viz


# ============================================================
# STANDALONE DEMO
# ============================================================

def demo_static():
    print("🤖 RoArm-M2-S 3D Visualisierung - Statische Demo")
    poses = [
        ("Home-Position", 0.0, 0.0, 90.0, 180.0),
        ("Base links", -45.0, 0.0, 90.0, 180.0),
        ("Arm hoch", 0.0, 45.0, 90.0, 180.0),
        ("Greifen (unten)", 0.0, -15.0, 30.0, 180.0),
    ]
    for name, b, s, e, h in poses:
        print(f"   Zeige: {name}")
        viz = RobotVisualizer()
        viz.show_pose(b, s, e, h)


def demo_animation():
    print("🤖 RoArm-M2-S 3D Visualisierung - Animation Demo")
    waypoints = []
    for i in range(100):
        t = i / 100.0 * 2 * math.pi
        waypoints.append({
            "b": 45.0 * math.sin(t),
            "s": 15.0 * math.sin(t * 2),
            "e": 90.0 + 30.0 * math.cos(t),
            "h": 180.0,
        })
    viz = RobotVisualizer()
    viz.show_trajectory(waypoints, fps=30)


def demo_live():
    """Demonstriert den Live-Modus (separater Prozess)."""
    print("🤖 RoArm-M2-S 3D Visualisierung - Live Demo")
    print("   Sendet Posen an separaten Visualizer-Prozess.\n")

    viz = RobotVisualizer(live=True)
    viz.start()
    time.sleep(1.0)  # Warten bis Fenster offen

    try:
        for i in range(200):
            t = i / 200.0 * 2 * math.pi
            b = 45.0 * math.sin(t)
            s = 15.0 * math.sin(t * 2)
            e = 90.0 + 30.0 * math.cos(t)
            h = 180.0

            viz.update_pose(b, s, e, h)
            time.sleep(0.05)

            if not viz.is_running:
                print("   Fenster geschlossen.")
                break

        print("   Demo fertig. Fenster schließt in 3s...")
        time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n   Abgebrochen.")
    finally:
        viz.stop()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RoArm-M2-S 3D Visualisierung")
    p.add_argument("--mode", choices=["static", "animation", "live"],
                   default="static")
    p.add_argument("--pose", type=str, default=None,
                   help="Einzelne Pose: 'b=0,s=0,e=90,h=180'")
    args = p.parse_args()

    if args.pose:
        pose = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        for part in args.pose.split(","):
            k, v = part.strip().split("=")
            pose[k.strip()] = float(v.strip())
        viz = RobotVisualizer()
        viz.show_pose(**pose)
    elif args.mode == "static":
        demo_static()
    elif args.mode == "animation":
        demo_animation()
    elif args.mode == "live":
        demo_live()
