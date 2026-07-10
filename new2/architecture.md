roarm_lib/
├── __init__.py              # Re-exports alles
├── hardware.py              # Arm-Kommunikation (aus roarm_m2s.py extrahiert)
├── vision.py                # YOLO-Wrapper, Bild→BBox, JPG-Export
├── dsl.py                   # Parser + Interpreter für die Skriptsprache
├── recorder.py              # Aufzeichnung → DSL-Dateien + Bilder
└── policy.py                # NN das nur BBoxes sieht, Training

scripts/
├── record.py                # Aufzeichnen (mit/ohne YOLO)
├── play.py                  # DSL-Skript abspielen (Step-by-Step)
├── train.py                 # Modell trainieren
├── extract_images.py        # JPGs aus Recordings extrahieren
└── run_policy.py            # Trainiertes Modell ausführen
