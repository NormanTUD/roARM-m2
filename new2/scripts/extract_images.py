#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
# ]
# ///

import os
import sys

def _ensure_uv():
    if os.environ.get("_UV_SAFE_ENV") == "1":
        return
    os.environ["_UV_SAFE_ENV"] = "1"
    from datetime import datetime, timedelta, timezone
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

"""
Image Extractor — Extrahiert JPGs aus Recordings für YOLO-Annotation.

Liest aufgezeichnete Episoden (Video oder Kamera-Frames) und speichert
jeden N-ten Frame als JPG in einem Ordner, der direkt in ein
YOLO-Annotation-Tool (z.B. CVAT, Roboflow, Label Studio) importiert werden kann.

Usage:
    python3 extract_images.py recordings/
    python3 extract_images.py recordings/ --every 5 --output yolo_images/
    python3 extract_images.py recordings/ --from-video
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("ERROR: opencv-python required! pip install opencv-python")
    sys.exit(1)


def extract_from_videos(recordings_dir: Path, output_dir: Path,
                        every_n: int = 5, max_per_episode: int = 200) -> int:
    """Extrahiert Frames aus MP4-Videos (LeRobot-Format)."""
    video_dir = recordings_dir / "videos" / "chunk-000" / "observation.images.top"
    if not video_dir.exists():
        # Versuche alternatives Layout
        video_dir = recordings_dir / "lerobot_dataset" / "videos" / "chunk-000" / "observation.images.top"

    if not video_dir.exists():
        print(f"  ⚠ Kein Video-Verzeichnis gefunden: {video_dir}")
        return 0

    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"  ⚠ Keine MP4-Dateien in {video_dir}")
        return 0

    total_saved = 0

    for video_path in video_files:
        ep_name = video_path.stem  # z.B. "episode_000000"
        ep_dir = output_dir / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ⚠ Kann nicht öffnen: {video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  📹 {video_path.name}: {frame_count} Frames @ {fps:.0f} FPS")

        saved_in_ep = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0 and saved_in_ep < max_per_episode:
                filename = ep_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved_in_ep += 1
                total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"    → {saved_in_ep} Bilder gespeichert")

    return total_saved


def extract_from_camera_live(output_dir: Path, camera_index: int = 2,
                             num_images: int = 50, interval_sec: float = 0.5) -> int:
    """
    Nimmt live Bilder von der Kamera auf (für initiale Annotation).
    Drücke SPACE zum Speichern, Q zum Beenden.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # Fallback
        for idx in [0, 2, 1, 4]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                break
        if not cap.isOpened():
            print("  ✗ Keine Kamera gefunden!")
            return 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n  📷 Live-Capture Modus")
    print(f"     SPACE = Bild speichern")
    print(f"     A     = Auto-Capture ({interval_sec}s Intervall)")
    print(f"     Q     = Beenden")
    print(f"     Ziel: {output_dir}")

    saved = 0
    auto_capture = False
    last_auto_time = 0

    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        # Display
        display = frame.copy()
        status = f"Gespeichert: {saved}/{num_images}"
        if auto_capture:
            status += " [AUTO]"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=Save  A=Auto  Q=Quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("Image Extractor", display)

        key = cv2.waitKey(30) & 0xFF

        should_save = False

        if key == ord(' '):
            should_save = True
        elif key == ord('a'):
            auto_capture = not auto_capture
            print(f"    Auto-Capture: {'AN' if auto_capture else 'AUS'}")
        elif key == ord('q'):
            break

        # Auto-capture
        if auto_capture and (time.time() - last_auto_time) >= interval_sec:
            should_save = True
            last_auto_time = time.time()

        if should_save:
            timestamp = int(time.time() * 1000)
            filename = output_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
            print(f"    📷 [{saved}] {filename.name}")

    cap.release()
    cv2.destroyAllWindows()
    return saved


def extract_from_episode_json(recordings_dir: Path, output_dir: Path,
                              every_n: int = 10) -> int:
    """
    Extrahiert Bilder aus JSON-Episoden die Detections enthalten.
    Erzeugt auch YOLO-Format .txt Annotation-Dateien wenn Detections vorhanden.
    """
    json_files = sorted(recordings_dir.glob("episode_*.json"))
    if not json_files:
        # Versuche LeRobot-Unterverzeichnis
        json_files = sorted(
            (recordings_dir / "lerobot_dataset" / "data" / "chunk-000").glob("episode_*.json")
        )

    if not json_files:
        print(f"  ⚠ Keine Episode-JSONs gefunden in {recordings_dir}")
        return 0

    # Wir können aus JSONs keine Bilder extrahieren (nur Metadaten),
    # aber wir können die Detection-Daten als YOLO-Annotations exportieren
    annotations_dir = output_dir / "labels"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    total_annotations = 0

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        frames = data.get("frames", data if isinstance(data, list) else [])

        for i, frame in enumerate(frames):
            if i % every_n != 0:
                continue

            detections = frame.get("detections", [])
            if not detections:
                continue

            # YOLO-Format: class_id cx cy w h (normalisiert)
            # Wir kennen die class_ids nicht genau, also nutzen wir class_name → Index
            ann_filename = annotations_dir / f"{json_path.stem}_frame_{i:06d}.txt"
            with open(ann_filename, 'w') as f:
                for det in detections:
                    bbox = det.get("target_bbox_normalized",
                                   det.get("bbox", [0, 0, 0, 0]))
                    if len(bbox) == 4:
                        if all(0 <= v <= 1 for v in bbox):
                            # Bereits normalisiert (x1, y1, x2, y2)
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            w = bbox[2] - bbox[0]
                            h = bbox[3] - bbox[1]
                        else:
                            # Pixel-Werte → normalisieren (640x480)
                            cx = (bbox[0] + bbox[2]) / 2 / 640
                            cy = (bbox[1] + bbox[3]) / 2 / 480
                            w = (bbox[2] - bbox[0]) / 640
                            h = (bbox[3] - bbox[1]) / 480

                        class_name = det.get("class", "object")
                        # Einfach class_id = 0 für jetzt
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        total_annotations += 1

    if total_annotations > 0:
        print(f"  ✓ {total_annotations} YOLO-Annotations exportiert → {annotations_dir}")
    return total_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Extrahiert Bilder aus Recordings für YOLO-Annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Aus Videos extrahieren (jeden 5. Frame)
  python3 extract_images.py recordings/ --from-video --every 5

  # Live von Kamera aufnehmen
  python3 extract_images.py --live --camera 2 --output yolo_images/

  # Annotations aus JSON-Episoden exportieren
  python3 extract_images.py recordings/ --annotations
        """
    )

    parser.add_argument("recordings_dir", type=str, nargs="?", default="recordings",
                        help="Recordings-Verzeichnis")
    parser.add_argument("--output", type=str, default="yolo_training_images",
                        help="Output-Verzeichnis für Bilder")
    parser.add_argument("--every", type=int, default=5,
                        help="Jeden N-ten Frame extrahieren")
    parser.add_argument("--from-video", action="store_true",
                        help="Aus MP4-Videos extrahieren")
    parser.add_argument("--live", action="store_true",
                        help="Live von Kamera aufnehmen")
    parser.add_argument("--camera", type=int, default=2,
                        help="Kamera-Index für Live-Modus")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Max. Bilder im Live-Modus")
    parser.add_argument("--annotations", action="store_true",
                        help="YOLO-Annotations aus JSON exportieren")
    parser.add_argument("--max-per-episode", type=int, default=200,
                        help="Max. Bilder pro Episode")

    args = parser.parse_args()

    recordings_path = Path(args.recordings_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  🖼️  YOLO Image Extractor")
    print(f"{'='*50}")
    print(f"  Input:  {recordings_path}")
    print(f"  Output: {output_path}")

    total = 0

    if args.live:
        total = extract_from_camera_live(
            output_path, camera_index=args.camera,
            num_images=args.num_images,
        )
    elif args.from_video:
        total = extract_from_videos(
            recordings_path, output_path,
            every_n=args.every, max_per_episode=args.max_per_episode,
        )
    elif args.annotations:
        total = extract_from_episode_json(
            recordings_path, output_path, every_n=args.every,
        )
    else:
        # Default: versuche Videos, dann Live
        total = extract_from_videos(
            recordings_path, output_path,
            every_n=args.every, max_per_episode=args.max_per_episode,
        )
        if total == 0:
            print("\n  Keine Videos gefunden. Starte Live-Capture...")
            total = extract_from_camera_live(
                output_path, camera_index=args.camera,
                num_images=args.num_images,
            )

    print(f"\n  ✓ Fertig! {total} Bilder/Annotations extrahiert")
    print(f"  → Importiere {output_path} in dein YOLO-Annotation-Tool")
    print(f"  → Trainiere dann: yolo train data=dataset.yaml model=yolo11n.pt epochs=50")


if __name__ == "__main__":
    main()

