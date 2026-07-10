"""
Policy-Modul: Neuronales Netz das NUR Bounding Boxes sieht.

Kernidee (Faserbündel-Analogie):
- Die Basis-Mannigfaltigkeit ist der YOLO-BBox-Raum (Position/Größe der Objekte)
- Die Faser darüber ist der Raum der nützlichen Bewegungen
- Wenn die Position im Raum bekannt ist (via BBoxes), kann die gesamte
  Bewegungs-Mannigfaltigkeit invariant transformiert werden
- → Viel schnelleres Lernen weil keine irrelevanten Pixel-Daten

Input: [bbox1_cx, bbox1_cy, bbox1_w, bbox1_h, bbox1_conf, ..., arm_state]
Output: [action_chunk]

Das NN sieht NIEMALS rohe Bilder — nur die abstrahierten BBox-Koordinaten.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BBoxObservation:
    """
    Konvertiert YOLO-Detections in einen festen Vektor für das NN.

    Format pro Objekt: [cx, cy, w, h, confidence] (normalisiert 0-1)
    Feste Anzahl Slots (zero-padded wenn weniger Objekte erkannt).

    Zusätzlich: Arm-State [base, shoulder, elbow, hand, gripper, led]
    """

    def __init__(self, max_objects: int = 5, num_classes: int = 10):
        self.max_objects = max_objects
        self.num_classes = num_classes
        # Pro Objekt: cx, cy, w, h, conf + one-hot class
        self.per_object_dim = 5 + num_classes
        self.bbox_dim = max_objects * self.per_object_dim
        self.arm_state_dim = 6  # base, shoulder, elbow, hand, gripper, led
        self.total_dim = self.bbox_dim + self.arm_state_dim

    def encode(self, detections: list, arm_state: list,
               class_names: List[str] = None) -> np.ndarray:
        """
        Kodiert Detections + Arm-State in einen flachen Vektor.

        Args:
            detections: Liste von Detection-Objekten oder Dicts
            arm_state: [base, shoulder, elbow, hand, gripper, led]
            class_names: Bekannte Klassen für One-Hot Encoding

        Returns:
            Flacher Vektor der Länge self.total_dim
        """
        vec = np.zeros(self.total_dim, dtype=np.float32)

        # BBox-Teil
        for i in range(min(len(detections), self.max_objects)):
            det = detections[i]
            offset = i * self.per_object_dim

            if hasattr(det, 'bbox'):
                # Detection-Objekt
                vec[offset] = det.bbox.x_center
                vec[offset + 1] = det.bbox.y_center
                vec[offset + 2] = det.bbox.width
                vec[offset + 3] = det.bbox.height
                vec[offset + 4] = det.confidence
                cls_name = det.class_name
            elif isinstance(det, dict):
                # Dict-Format
                bbox = det.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    # Normalisiert oder Pixel?
                    if all(0 <= v <= 1 for v in bbox):
                        vec[offset] = (bbox[0] + bbox[2]) / 2
                        vec[offset + 1] = (bbox[1] + bbox[3]) / 2
                        vec[offset + 2] = bbox[2] - bbox[0]
                        vec[offset + 3] = bbox[3] - bbox[1]
                    else:
                        # Pixel → normalisieren (640x480 angenommen)
                        vec[offset] = (bbox[0] + bbox[2]) / 2 / 640
                        vec[offset + 1] = (bbox[1] + bbox[3]) / 2 / 480
                        vec[offset + 2] = (bbox[2] - bbox[0]) / 640
                        vec[offset + 3] = (bbox[3] - bbox[1]) / 480
                vec[offset + 4] = det.get('confidence', 0.0)
                cls_name = det.get('class', '')
            else:
                continue

            # One-Hot Class
            if class_names and cls_name in class_names:
                cls_idx = class_names.index(cls_name)
                if cls_idx < self.num_classes:
                    vec[offset + 5 + cls_idx] = 1.0

        # Arm-State-Teil (normalisiert)
        arm_offset = self.bbox_dim
        if len(arm_state) >= 6:
            vec[arm_offset] = arm_state[0] / 90.0      # base: -90..90 → -1..1
            vec[arm_offset + 1] = arm_state[1] / 60.0  # shoulder: -30..60
            vec[arm_offset + 2] = arm_state[2] / 180.0 # elbow: 0..180 → 0..1
            vec[arm_offset + 3] = arm_state[3] / 270.0 # hand: 0..270 → 0..1
            vec[arm_offset + 4] = arm_state[4]          # gripper: 0 or 1
            vec[arm_offset + 5] = arm_state[5]          # led: 0..1

        return vec


class BBoxPolicy(nn.Module):
    """
    Policy-Netzwerk das NUR BBox-Koordinaten + Arm-State sieht.

    Architektur:
    - Input: [max_objects * (5 + num_classes) + 6] (BBoxes + Arm)
    - Hidden: Transformer-Encoder (lernt Beziehungen zwischen Objekten)
    - Output: Action-Chunk [chunk_size, 6]

    Vorteile gegenüber Bild-basiertem Ansatz:
    - Extrem kleiner Input (z.B. 81 statt 640*480*3 = 921600)
    - Invariant gegenüber Textur, Beleuchtung, Hintergrund
    - Lernt nur räumliche Beziehungen
    - Trainiert in Minuten statt Stunden
    """

    def __init__(self, max_objects: int = 5, num_classes: int = 10,
                 arm_state_dim: int = 6, action_dim: int = 6,
                 chunk_size: int = 10, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()

        if not HAS_TORCH:
            raise ImportError("PyTorch benötigt: pip install torch")

        self.max_objects = max_objects
        self.num_classes = num_classes
        self.per_object_dim = 5 + num_classes
        self.arm_state_dim = arm_state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        input_dim = max_objects * self.per_object_dim + arm_state_dim

        # Encoder
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Object-Attention (lernt welche Objekte relevant sind)
        self.object_encoder = nn.Sequential(
            nn.Linear(self.per_object_dim, hidden_dim),
            nn.ReLU(),
        )

        self.arm_encoder = nn.Sequential(
            nn.Linear(arm_state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Cross-Attention: Arm-State attended auf Objekte
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action-Chunk Decoder
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: [B, input_dim] — BBoxes + Arm-State

        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = observation.shape[0]

        # Split in Objekte und Arm-State
        bbox_flat = observation[:, :self.max_objects * self.per_object_dim]
        arm_state = observation[:, self.max_objects * self.per_object_dim:]

        # Objekte einzeln kodieren
        objects = bbox_flat.view(B, self.max_objects, self.per_object_dim)
        obj_features = self.object_encoder(objects)  # [B, max_objects, hidden]

        # Arm-State kodieren
        arm_features = self.arm_encoder(arm_state).unsqueeze(1)  # [B, 1, hidden]

        # Zusammen als Sequenz für Transformer
        sequence = torch.cat([arm_features, obj_features], dim=1)  # [B, 1+max_objects, hidden]

        # Self-Attention (lernt Beziehungen)
        encoded = self.transformer(sequence)  # [B, 1+max_objects, hidden]

        # Memory für Decoder (alle encoded tokens)
        memory = encoded

        # Action-Chunk dekodieren
        query = self.pos_embed.expand(B, -1, -1)  # [B, chunk_size, hidden]
        decoded = self.decoder(query, memory)  # [B, chunk_size, hidden]

        actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
        return actions

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Inference: Einzelne Observation → Action-Chunk."""
        self.eval()
        device = next(self.parameters()).device
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(device)
        actions = self.forward(obs_tensor)
        return actions.squeeze(0).cpu().numpy()


class BBoxDataset(Dataset):
    """
    Dataset das aus DSL-Recordings + YOLO-Detections besteht.
    Jeder Sample: (bbox_observation, action_chunk)
    """

    def __init__(self, recordings_dir: Path, chunk_size: int = 10,
                 max_objects: int = 5, num_classes: int = 10,
                 class_names: List[str] = None):
        self.chunk_size = chunk_size
        self.encoder = BBoxObservation(max_objects, num_classes)
        self.class_names = class_names or []
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        self._load_recordings(recordings_dir)

    def _load_recordings(self, recordings_dir: Path):
        """Lädt alle Recordings und baut Samples."""
        recordings_dir = Path(recordings_dir)

        # JSON-Episoden laden
        for ep_file in sorted(recordings_dir.glob("episode_*.json")):
            self._load_episode_json(ep_file)

        # LeRobot-Format
        lerobot_data = recordings_dir / "lerobot_dataset" / "data" / "chunk-000"
        if lerobot_data.exists():
            for ep_file in sorted(lerobot_data.glob("episode_*.json")):
                self._load_episode_json(ep_file)

        print(f"[BBoxDataset] {len(self.samples)} Samples geladen")

    def _load_episode_json(self, path: Path):
        """Lädt eine Episode und extrahiert Samples."""
        with open(path) as f:
            data = json.load(f)

        frames = data.get("frames", data if isinstance(data, list) else [])
        if not frames:
            return

        for i in range(len(frames) - self.chunk_size):
            frame = frames[i]

            # Observation bauen
            detections = frame.get("detections", [])
            arm = frame.get("arm_state", {})
            led = frame.get("led_brightness", 255) / 255.0

            arm_vec = [
                arm.get("base_deg", 0),
                arm.get("shoulder_deg", 0),
                arm.get("elbow_deg", 90),
                arm.get("hand_deg", 180),
                0.0 if arm.get("gripper_open", True) else 1.0,
                led,
            ]

            obs = self.encoder.encode(detections, arm_vec, self.class_names)

            # Action-Chunk (nächste chunk_size Frames)
            action_chunk = []
            for j in range(self.chunk_size):
                future = frames[i + j]
                future_arm = future.get("arm_state", {})
                future_led = future.get("led_brightness", 255) / 255.0
                action = [
                    future_arm.get("base_deg", 0) / 90.0,
                    future_arm.get("shoulder_deg", 0) / 60.0,
                    future_arm.get("elbow_deg", 90) / 180.0,
                    future_arm.get("hand_deg", 180) / 270.0,
                    0.0 if future_arm.get("gripper_open", True) else 1.0,
                    future_led,
                ]
                action_chunk.append(action)

            action_chunk = np.array(action_chunk, dtype=np.float32)
            self.samples.append((obs, action_chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, action = self.samples[idx]
        return {
            "observation": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
        }


def train_policy(recordings_dir: str, output_dir: str = "trained_bbox_policy",
                 epochs: int = 200, batch_size: int = 32, lr: float = 1e-4,
                 chunk_size: int = 10, max_objects: int = 5,
                 class_names: List[str] = None):
    """
    Trainiert eine BBox-Policy auf Recordings.

    Deutlich schneller als Bild-basiertes Training weil:
    - Input ist ~80 Dimensionen statt ~900000 (Bild)
    - Kein CNN nötig
    - Konvergiert in Minuten
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch benötigt!")

    recordings_path = Path(recordings_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  BBox-Policy Training")
    print(f"{'='*50}")
    print(f"  Recordings: {recordings_path}")
    print(f"  Output:     {output_path}")
    print(f"  Epochs:     {epochs}")
    print(f"  Chunk Size: {chunk_size}")

    # Dataset
    dataset = BBoxDataset(
        recordings_path, chunk_size=chunk_size,
        max_objects=max_objects, class_names=class_names,
    )

    if len(dataset) == 0:
        print("  ✗ Keine Samples gefunden!")
        return None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Model
    num_classes = len(class_names) if class_names else 10
    model = BBoxPolicy(
        max_objects=max_objects, num_classes=num_classes,
        chunk_size=chunk_size, hidden_dim=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model:      {num_params:,} Parameter")
    print(f"  Device:     {device}")
    print(f"  Samples:    {len(dataset)}")
    print()

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            obs = batch["observation"].to(device)
            action = batch["action"].to(device)

            optimizer.zero_grad()
            pred = model(obs)
            loss = nn.functional.mse_loss(pred, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "max_objects": max_objects,
                    "num_classes": num_classes,
                    "chunk_size": chunk_size,
                    "hidden_dim": 128,
                    "class_names": class_names,
                },
            }, output_path / "bbox_policy_best.pt")

        if (epoch + 1) % max(1, epochs // 20) == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\n  ✓ Training fertig! Best Loss: {best_loss:.6f}")
    print(f"  ✓ Modell: {output_path / 'bbox_policy_best.pt'}")

    return model

