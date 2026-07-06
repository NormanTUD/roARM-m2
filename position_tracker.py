# Neues File: position_tracker.py
"""
Trackt die Arm-Position basierend auf gesendeten Befehlen.
Fallback wenn get_status() nicht antwortet.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackedPosition:
    """Geschätzte Position basierend auf gesendeten Befehlen."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    base_deg: float = 0.0
    shoulder_deg: float = 0.0
    elbow_deg: float = 0.0
    hand_deg: float = 180.0


class PositionTracker:
    """
    Trackt die Arm-Position ohne Feedback.
    
    RoArm-M2-S Kinematik (vereinfacht):
    - Link 1 (Shoulder→Elbow): ~128mm
    - Link 2 (Elbow→Wrist): ~128mm  
    - Base dreht um Z-Achse
    
    WICHTIG: Diese Werte müssen für deinen Arm kalibriert werden!
    """
    
    # Arm-Dimensionen (mm) – KALIBRIEREN!
    L1 = 128.0  # Shoulder → Elbow
    L2 = 128.0  # Elbow → Wrist/Gripper
    BASE_HEIGHT = 72.0  # Höhe der Base über Tisch
    
    def __init__(self):
        self.pos = TrackedPosition()
        self._last_cartesian: Optional[Tuple[float, float, float]] = None
    
    def update_from_joints_degrees(self, b: float, s: float, e: float, h: float):
        """Update Position nach Joint-Befehl."""
        self.pos.base_deg = b
        self.pos.shoulder_deg = s
        self.pos.elbow_deg = e
        self.pos.hand_deg = h
        self._compute_cartesian()
    
    def update_from_cartesian(self, x: float, y: float, z: float):
        """Update Position nach Cartesian-Befehl."""
        self._last_cartesian = (x, y, z)
        self.pos.x = x
        self.pos.y = y
        self.pos.z = z
        # Rückwärts-Kinematik für Winkel (vereinfacht)
        self.pos.base_deg = math.degrees(math.atan2(y, x)) if (x != 0 or y != 0) else 0
    
    def _compute_cartesian(self):
        """Forward Kinematics: Joints → XYZ."""
        b_rad = math.radians(self.pos.base_deg)
        s_rad = math.radians(self.pos.shoulder_deg)
        e_rad = math.radians(self.pos.elbow_deg)
        
        # Vereinfachte 2-Link Planar Kinematik in der Vertikalebene
        # Dann um Base-Achse rotiert
        
        # Shoulder-Winkel: 0° = horizontal nach vorne
        # Elbow-Winkel: relativ zum Shoulder-Link
        r = self.L1 * math.cos(s_rad) + self.L2 * math.cos(s_rad + e_rad - math.pi/2)
        z = self.BASE_HEIGHT + self.L1 * math.sin(s_rad) + self.L2 * math.sin(s_rad + e_rad - math.pi/2)
        
        self.pos.x = r * math.cos(b_rad)
        self.pos.y = r * math.sin(b_rad)
        self.pos.z = z
        
        if self._last_cartesian is None:
            self._last_cartesian = (self.pos.x, self.pos.y, self.pos.z)
    
    @property
    def cartesian(self) -> Tuple[float, float, float]:
        return (self.pos.x, self.pos.y, self.pos.z)
