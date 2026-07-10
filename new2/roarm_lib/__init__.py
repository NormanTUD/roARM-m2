"""
RoArm-M2-S Library
==================
Modulare Bibliothek für Roboterarm-Steuerung mit DSL und YOLO-Vision.
"""

from .hardware import RoArmHardware, ArmState
from .vision import VisionSystem, Detection, BoundingBox
from .dsl import RoArmDSL, DSLInterpreter, DSLRecorder
from .recorder import SessionRecorder
from .policy import BBoxPolicy, train_policy

__all__ = [
    "RoArmHardware", "ArmState",
    "VisionSystem", "Detection", "BoundingBox",
    "RoArmDSL", "DSLInterpreter", "DSLRecorder",
    "SessionRecorder",
    "BBoxPolicy", "train_policy",
]

