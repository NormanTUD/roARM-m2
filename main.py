#!/usr/bin/env python3
"""RoArm-M2-S Controller — Minimal Entry Point"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="RoArm-M2-S Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    parser.add_argument("--target", type=str, default=None, help="Ziel-Objekt zum Greifen")
    parser.add_argument("--scan", action="store_true", help="Nur Live-Stream zeigen")
    parser.add_argument("--camera", type=int, default=None, help="Kamera-Index")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO-Modell")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min. Confidence")
    parser.add_argument("--headless", action="store_true", help="Kein GUI-Fenster")
    parser.add_argument("--no-vision", action="store_true", help="Vision deaktivieren")
    parser.add_argument("--debug", action="store_true", help="Verbose Debug-Ausgaben")
    args = parser.parse_args()

    from roarm_m2s import RoArmM2S

    if args.target or args.scan:
        from eye_in_hand import EyeInHandController

        arm = RoArmM2S(port=args.port, enable_vision=False)
        controller = EyeInHandController(
            arm=arm,
            camera_index=args.camera,
            model_path=args.model,
            confidence=args.confidence,
            headless=args.headless,
            debug=args.debug,
        )

        try:
            if args.target:
                controller.grab(args.target)
            elif args.scan:
                controller.live_scan()
        finally:
            controller.shutdown()
            arm.disconnect()
    else:
        with RoArmM2S(port=args.port, enable_vision=False) as arm:
            from eye_in_hand import run_test_demo
            run_test_demo(arm)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\n[FEHLER] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Abgebrochen]")
        sys.exit(0)
