#!/usr/bin/env python3
"""dashboard.py - RoArm-M2-S Unified TUI Dashboard v2

Tabs:
- Tab 1: TEACH (Recording mit Live-Feedback)
- Tab 2: PLAY (Recordings abspielen)
- Tab 3: CALIBRATE (Kalibrierung starten/verwalten)
- Tab 4: SERVO (Einzelne Servos ansteuern/auslesen)

Alle Aktionen per Keyboard Shortcuts.
Auto-Connect wenn USB-Port gefunden.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "numpy",
#     "scipy",
#     "textual>=0.79.0",
#     "matplotlib",
#     "rich",
#     "Pillow",
#     "pyyaml",
#     "psutil",
# ]
# ///

try:
    import os
    os.environ["TEXTUAL_RUNNING"] = "1"

    import sys

    from bootstrap import ensure_uv
    ensure_uv()

    from dashboard_core.app import RoArmDashboard


    def main():
        app = RoArmDashboard()
        app.run()

    if __name__ == "__main__":
            main()
except KeyboardInterrupt:
    sys.exit(0)
