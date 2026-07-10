#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ultralytics",
#     "opencv-python",
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
YOLO Training Helper — Trainiert ein YOLO-Modell auf deinen annotierten Bildern.

Workflow:
1. Bilder extrahieren: python3 extract_images.py --live
2. Annotieren in deinem YOLO-Tool (Roboflow, CVAT, Label Studio)
3. Dataset exportieren als YOLO-Format (images/ + labels/ + data.yaml)
4. Trainieren: python3 train_yolo.py path/to/data.yaml

Das trainierte Modell wird dann im Recorder und Policy-Runner verwendet.

Usage:
    python3 train_yolo.py dataset/data.yaml
    python3 train_yolo.py dataset/data.yaml --epochs 100 --model yolo11n.pt
    python3 train_yolo.py dataset/data.yaml --imgsz 640 --batch 8
"""

import argparse
from pathlib import Path


def create_dataset_yaml(images_dir: str, classes: list, output_path: str = "dataset.yaml"):
    """
    Erstellt eine data.yaml für YOLO-Training wenn du nur einen Ordner mit
    Bildern und Labels hast.
    """
    images_path = Path(images_dir)
    labels_path = images_path.parent / "labels"

    yaml_content = f"""# Auto-generated YOLO dataset config
path: {images_path.parent.absolute()}
train: {images_path.name}
val: {images_path.name}

nc: {len(classes)}
names: {classes}
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"  ✓ Dataset YAML erstellt: {output_path}")
    print(f"    Klassen: {classes}")
    print(f"    Bilder:  {images_path}")
    print(f"    Labels:  {labels_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Training für RoArm Objekt-Erkennung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard-Training
  python3 train_yolo.py my_dataset/data.yaml

  # Mehr Epochen, größere Bilder
  python3 train_yolo.py my_dataset/data.yaml --epochs 100 --imgsz 640

  # Dataset-YAML aus Ordner erstellen
  python3 train_yolo.py --create-yaml images/ --classes charger wall_marker table

  # Kleines Modell für Raspberry Pi
  python3 train_yolo.py my_dataset/data.yaml --model yolo11n.pt
        """
    )

    parser.add_argument("data_yaml", type=str, nargs="?", default=None,
                        help="Pfad zur data.yaml (YOLO-Format)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Basis-Modell (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Trainings-Epochen (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Bildgröße (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch-Größe (default: 16)")
    parser.add_argument("--output", type=str, default="yolo_trained",
                        help="Output-Verzeichnis")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (0=GPU, cpu=CPU, auto)")

    # Dataset-YAML erstellen
    parser.add_argument("--create-yaml", type=str, default=None,
                        help="Erstellt data.yaml aus Bilder-Ordner")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Klassen-Namen für --create-yaml")

    args = parser.parse_args()

    # Dataset-YAML erstellen
    if args.create_yaml:
        if not args.classes:
            print("✗ --classes benötigt wenn --create-yaml verwendet wird!")
            print("  Beispiel: --create-yaml images/ --classes charger wall_marker")
            sys.exit(1)
        yaml_path = create_dataset_yaml(args.create_yaml, args.classes)
        if not args.data_yaml:
            args.data_yaml = yaml_path

    if not args.data_yaml:
        print("✗ Bitte data.yaml angeben!")
        print("  python3 train_yolo.py my_dataset/data.yaml")
        print("  oder: python3 train_yolo.py --create-yaml images/ --classes obj1 obj2")
        sys.exit(1)

    data_path = Path(args.data_yaml)
    if not data_path.exists():
        print(f"✗ Datei nicht gefunden: {data_path}")
        sys.exit(1)

    # Training
    print(f"\n{'='*50}")
    print(f"  🎯 YOLO Training")
    print(f"{'='*50}")
    print(f"  Dataset:  {data_path}")
    print(f"  Modell:   {args.model}")
    print(f"  Epochen:  {args.epochs}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Output:   {args.output}")
    print()

    try:
        from ultralytics import YOLO

        model = YOLO(args.model)

        results = model.train(
            data=str(data_path.absolute()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.output,
            name="train",
            exist_ok=True,
            device=args.device,
            verbose=True,
        )

        # Best model Pfad
        best_model = Path(args.output) / "train" / "weights" / "best.pt"
        if best_model.exists():
            print(f"\n  ✓ Training fertig!")
            print(f"  ✓ Bestes Modell: {best_model}")
            print(f"\n  Nächste Schritte:")
            print(f"    # Im Recorder verwenden:")
            print(f"    python3 record.py --model {best_model}")
            print(f"    # In der Policy verwenden:")
            print(f"    python3 run_policy.py model.pt --yolo {best_model}")
        else:
            print(f"\n  ✓ Training fertig! Ergebnisse in: {args.output}/train/")

    except ImportError:
        print("✗ ultralytics nicht installiert!")
        print("  pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Training-Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
