"""
RoArm-M2-S Library
==================
Modulare Bibliothek für Roboterarm-Steuerung mit DSL und YOLO-Vision.
"""

from .hardware import RoArmHardware, ArmState
from .vision import VisionSystem, Detection, BoundingBox
from .dsl import DSLParser, DSLInterpreter, DSLRecorder
from .recorder import SessionRecorder
from .policy import BBoxPolicy, BBoxObservation, train_policy

__all__ = [
    "RoArmHardware", "ArmState",
    "VisionSystem", "Detection", "BoundingBox",
    "DSLParser", "DSLInterpreter", "DSLRecorder",
    "SessionRecorder",
    "BBoxPolicy", "BBoxObservation", "train_policy",
]
