#!/usr/bin/env python3
"""
Train Policy – Behaviour Cloning auf aufgezeichneten Teleop-Daten.

Architektur: MLP mit optionalem LSTM für Sequenz-Kontext.
Input:  BBox-Features (normalisiert) + aktuelle Arm-Gelenkwinkel
Output: Diskrete Aktion (welche Bewegung als nächstes)

Nutzung:
  python train_policy.py --data recordings/ --epochs 100
  python train_policy.py --data recordings/ --epochs 200 --seq-len 10 --use-lstm
  python train_policy.py --data recordings/ --target bottle --lr 0.001
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ─── Aktions-Encoding ─────────────────────────────────────────────────────────

ACTION_SPACE = [
    "",               # 0: keine Aktion (idle)
    "base_left",      # 1
    "base_right",     # 2
    "shoulder_up",    # 3
    "shoulder_down",  # 4
    "elbow_up",       # 5
    "elbow_down",     # 6
    "hand_left",      # 7
    "hand_right",     # 8
    "gripper_open",   # 9
    "gripper_close",  # 10
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SPACE)}
NUM_ACTIONS = len(ACTION_SPACE)


# ─── Feature-Extraktion ──────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Konfiguration für Feature-Extraktion."""
    max_detections: int = 5       # Max. Anzahl BBoxes als Input
    normalize_bbox: bool = True   # BBox-Koordinaten auf [0,1] normalisieren
    include_arm_state: bool = True
    include_gripper: bool = True
    include_rel_to_target: bool = True


def extract_frame_features(frame: Dict, config: FeatureConfig) -> np.ndarray:
    """
    Extrahiert Feature-Vektor aus einem aufgezeichneten Frame.

    Features pro Detection (max_detections × 6):
      - bbox_cx_norm (0-1)
      - bbox_cy_norm (0-1)
      - bbox_w_norm (0-1)
      - bbox_h_norm (0-1)
      - confidence (0-1)
      - is_present (0 oder 1)

    Arm-State (5):
      - base_deg / 90 (normalisiert auf ~[-1, 1])
      - shoulder_deg / 60
      - elbow_deg / 180
      - hand_deg / 270
      - gripper_open (0 oder 1)

    Relative zum Target (4):
      - offset_px_x / 320 (normalisiert)
      - offset_px_y / 240
      - target_size_w / 640
      - target_size_h / 480
    """
    features = []

    # ─── Detection Features ───
    detections = frame.get("detections", [])
    for i in range(config.max_detections):
        if i < len(detections):
            det = detections[i]
            bbox = det.get("bbox", [0, 0, 0, 0])
            # Normalisiere auf Bildgröße (angenommen 640x480)
            cx = (bbox[0] + bbox[2]) / 2.0 / 640.0
            cy = (bbox[1] + bbox[3]) / 2.0 / 480.0
            w = (bbox[2] - bbox[0]) / 640.0
            h = (bbox[3] - bbox[1]) / 480.0
            conf = det.get("confidence", 0.0)
            features.extend([cx, cy, w, h, conf, 1.0])
        else:
            # Padding: keine Detection
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ─── Arm State ───
    if config.include_arm_state:
        arm = frame.get("arm_state", {})
        features.append(arm.get("base_deg", 0.0) / 90.0)
        features.append(arm.get("shoulder_deg", 0.0) / 60.0)
        features.append(arm.get("elbow_deg", 90.0) / 180.0)
        features.append(arm.get("hand_deg", 180.0) / 270.0)
        if config.include_gripper:
            features.append(1.0 if arm.get("gripper_open", True) else 0.0)

    # ─── Relative zum Target ───
    if config.include_rel_to_target:
        rel = frame.get("rel_to_target", None)
        if rel:
            features.append(rel.get("offset_px_x", 0.0) / 320.0)
            features.append(rel.get("offset_px_y", 0.0) / 240.0)
            size = rel.get("target_size_px", [0, 0])
            if isinstance(size, list) and len(size) >= 2:
                features.append(size[0] / 640.0)
                features.append(size[1] / 480.0)
            else:
                features.append(0.0)
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def get_feature_dim(config: FeatureConfig) -> int:
    """Berechnet die Feature-Dimension."""
    dim = config.max_detections * 6  # BBox features
    if config.include_arm_state:
        dim += 4  # base, shoulder, elbow, hand
        if config.include_gripper:
            dim += 1
    if config.include_rel_to_target:
        dim += 4  # offset_x, offset_y, size_w, size_h
    return dim


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TeleopDataset(Dataset):
    """
    Dataset aus aufgezeichneten Episoden.
    Jeder Sample = (features, action_idx) oder (feature_sequence, action_idx).
    """

    def __init__(self, data_dir: str, feature_config: FeatureConfig,
                 seq_len: int = 1, target_class: str = None,
                 only_successful: bool = True, skip_idle: bool = True):
        self.feature_config = feature_config
        self.seq_len = seq_len
        self.samples: List[Tuple[np.ndarray, int]] = []

        data_path = Path(data_dir)
        episode_files = sorted(data_path.glob("episode_*.json"))

        if not episode_files:
            raise FileNotFoundError(f"Keine Episoden in '{data_dir}' gefunden!")

        print(f"\n{'='*60}")
        print(f"  Dataset laden: {data_dir}")
        print(f"{'='*60}")

        total_frames = 0
        used_frames = 0
        episodes_loaded = 0

        for ep_file in episode_files:
            with open(ep_file, 'r') as f:
                episode = json.load(f)

            # Filter
            if only_successful and not episode.get("success", False):
                print(f"  ⊘ {ep_file.name} (nicht erfolgreich)")
                continue

            if target_class and episode.get("target_class", "") != target_class:
                print(f"  ⊘ {ep_file.name} (falsches Target: {episode.get('target_class')})")
                continue

            frames = episode.get("frames", [])
            total_frames += len(frames)

            if self.seq_len > 1:
                # Sequenz-Modus: Sliding Window
                self._add_sequences(frames, skip_idle)
            else:
                # Einzelframe-Modus
                self._add_single_frames(frames, skip_idle)

            episodes_loaded += 1
            print(f"  ✓ {ep_file.name} ({len(frames)} frames, "
                  f"{episode.get('duration_s', 0):.1f}s)")

        used_frames = len(self.samples)

        print(f"\n  Episoden geladen: {episodes_loaded}/{len(episode_files)}")
        print(f"  Frames total: {total_frames}")
        print(f"  Samples (nach Filter): {used_frames}")
        print(f"  Seq-Länge: {self.seq_len}")
        print(f"  Feature-Dim: {get_feature_dim(feature_config)}")

        # Aktions-Verteilung
        action_counts = {}
        for _, action_idx in self.samples:
            action_name = ACTION_SPACE[action_idx]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        print(f"\n  Aktions-Verteilung:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.samples) * 100
            bar = "█" * int(pct / 2)
            name = action if action else "(idle)"
            print(f"    {name:15s} {count:5d} ({pct:5.1f}%) {bar}")

        print(f"{'='*60}\n")

    def _add_single_frames(self, frames: List[Dict], skip_idle: bool):
        """Fügt einzelne Frames als Samples hinzu."""
        for frame in frames:
            action_str = frame.get("action", "")
            if skip_idle and action_str == "":
                continue

            action_idx = ACTION_TO_IDX.get(action_str, 0)
            features = extract_frame_features(frame, self.feature_config)
            self.samples.append((features, action_idx))

    def _add_sequences(self, frames: List[Dict], skip_idle: bool):
        """Fügt Sequenzen (Sliding Window) als Samples hinzu."""
        for i in range(len(frames) - self.seq_len + 1):
            window = frames[i:i + self.seq_len]
            # Aktion = letzte Aktion in der Sequenz
            action_str = window[-1].get("action", "")
            if skip_idle and action_str == "":
                continue

            action_idx = ACTION_TO_IDX.get(action_str, 0)

            # Feature-Sequenz
            seq_features = np.stack([
                extract_frame_features(f, self.feature_config)
                for f in window
            ])
            self.samples.append((seq_features, action_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, action_idx = self.samples[idx]
        return torch.FloatTensor(features), torch.LongTensor([action_idx]).squeeze()


# ─── Modell-Architekturen ─────────────────────────────────────────────────────

class GraspPolicyMLP(nn.Module):
    """
    Einfaches MLP für Behaviour Cloning.
    Input: Feature-Vektor (BBoxes + Arm-State)
    Output: Aktions-Wahrscheinlichkeiten
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 num_actions: int = NUM_ACTIONS, dropout: float = 0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GraspPolicyLSTM(nn.Module):
    """
    LSTM-basiertes Modell für Sequenz-Input.
    Nutzt zeitliche Korrelation zwischen Frames.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_actions: int = NUM_ACTIONS,
                 dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) für Sequenzen
           oder (batch, input_dim) für Einzelframes
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        # Nur letzten Zeitschritt nehmen
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        return self.output_head(last_hidden)


class GraspPolicyTransformer(nn.Module):
    """
    Kleiner Transformer für Sequenz-Input.
    Gut für längere Sequenzen und komplexere Muster.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 num_actions: int = NUM_ACTIONS, dropout: float = 0.1,
                 max_seq_len: int = 50):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        # CLS-Token-Stil: nehme letzten Output
        last = x[:, -1, :]
        return self.output_head(last)


# ─── Training ────────────────────────────────────────────────────────────────

class PolicyTrainer:
    """Trainiert die Greif-Policy."""

    def __init__(self, model: nn.Module, device: str = "auto",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 class_weights: Optional[np.ndarray] = None):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Class weights für unbalancierte Aktionen
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )

        print(f"\n  Device: {self.device}")
        print(f"  Parameter: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  LR: {lr}, Weight Decay: {weight_decay}")

    def compute_class_weights(self, dataset: TeleopDataset) -> np.ndarray:
        """Berechnet inverse Frequenz-Gewichte für unbalancierte Klassen."""
        counts = np.zeros(NUM_ACTIONS)
        for _, action_idx in dataset.samples:
            counts[action_idx] += 1

        # Inverse Frequenz, geglättet
        total = counts.sum()
        weights = np.where(counts > 0, total / (NUM_ACTIONS * counts), 1.0)
        # Clip extreme Gewichte
        weights = np.clip(weights, 0.1, 10.0)
        return weights

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, save_path: str = "policy_model.pt",
              patience: int = 20) -> Dict:
        """
        Trainiert das Modell.

        Returns:
            Dict mit Training-History.
        """
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"  Training starten ({epochs} Epochen)")
        print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            # ─── Train ───
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for features, actions in train_loader:
                features = features.to(self.device)
                actions = actions.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, actions)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == actions).sum().item()
                train_total += features.size(0)

            self.scheduler.step()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ─── Validation ───
            val_loss, val_acc = self._evaluate(val_loader)

            # ─── Logging ───
            current_lr = self.optimizer.param_groups[0]['lr']
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            # Progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
                      f"LR: {current_lr:.6f}")

            # ─── Best Model ───
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "action_space": ACTION_SPACE,
                    "feature_config": {
                        "max_detections": 5,
                        "normalize_bbox": True,
                        "include_arm_state": True,
                        "include_gripper": True,
                        "include_rel_to_target": True,
                    },
                    "model_class": self.model.__class__.__name__,
                    "input_dim": next(self.model.parameters()).shape[-1]
                                 if hasattr(next(self.model.parameters()), 'shape') else 0,
                }, save_path)
            else:
                patience_counter += 1

            # ─── Early Stopping ───
            if patience_counter >= patience:
                print(f"\n  ⚠ Early Stopping nach {epoch} Epochen (keine Verbesserung seit {patience})")
                break

        print(f"\n{'='*60}")
        print(f"  Training abgeschlossen!")
        print(f"  Bestes Modell: Epoch {best_epoch} | Val-Acc: {best_val_acc:.4f}")
        print(f"  Gespeichert: {save_path}")
        print(f"{'='*60}\n")

        return history

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluiert auf einem DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, actions in loader:
                features = features.to(self.device)
                actions = actions.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits, actions)

                total_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == actions).sum().item()
                total += features.size(0)

        return total_loss / total, correct / total


# ─── Confusion Matrix & Analyse ──────────────────────────────────────────────

def print_confusion_analysis(model: nn.Module, loader: DataLoader, device: torch.device):
    """Zeigt detaillierte Analyse der Vorhersagen."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(actions.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print(f"\n{'─'*60}")
    print(f"  Detaillierte Analyse (Val-Set)")
    print(f"{'─'*60}")

    # Per-Klasse Accuracy
    print(f"\n  {'Aktion':<16s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'─'*44}")

    for i, action_name in enumerate(ACTION_SPACE):
        mask = all_targets == i
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        acc = correct / total
        name = action_name if action_name else "(idle)"
        print(f"  {name:<16s} {correct:>8d} {total:>8d} {acc:>10.3f}")

    # Häufigste Verwechslungen
    print(f"\n  Häufigste Fehler:")
    confusion = {}
    for pred, target in zip(all_preds, all_targets):
        if pred != target:
            key = (ACTION_SPACE[target], ACTION_SPACE[pred])
            confusion[key] = confusion.get(key, 0) + 1

    sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    for (true_a, pred_a), count in sorted_conf:
        true_name = true_a if true_a else "(idle)"
        pred_name = pred_a if pred_a else "(idle)"
        print(f"    {true_name} → {pred_name}: {count}x")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Grasp Policy (Behaviour Cloning)")

    # Daten
    parser.add_argument("--data", type=str, default="recordings",
                        help="Verzeichnis mit Episode-JSONs")
    parser.add_argument("--target", type=str, default=None,
                        help="Nur Episoden mit diesem Target-Objekt")
    parser.add_argument("--include-failed", action="store_true",
                        help="Auch fehlgeschlagene Episoden nutzen")
    parser.add_argument("--include-idle", action="store_true",
                        help="Auch Idle-Frames (keine Aktion) nutzen")

    # Modell
    parser.add_argument("--arch", type=str, default="mlp",
                        choices=["mlp", "lstm", "transformer"],
                        help="Modell-Architektur")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32],
                        help="Hidden Layer Dimensionen (MLP)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden Dimension (LSTM/Transformer)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Anzahl Layers (LSTM/Transformer)")
    parser.add_argument("--seq-len", type=int, default=1,
                        help="Sequenz-Länge (>1 aktiviert Sequenz-Modus)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout-Rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Anzahl Epochen")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch-Größe")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight Decay")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation-Anteil")
    parser.add_argument("--patience", type=int, default=20, help="Early Stopping Patience")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Class Weights für unbalancierte Aktionen")

    # Features
    parser.add_argument("--max-detections", type=int, default=5,
                        help="Max. Anzahl Detections als Input")

    # Output
    parser.add_argument("--output", type=str, default="policy_model.pt",
                        help="Pfad für gespeichertes Modell")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")

    args = parser.parse_args()

    # ─── Feature Config ───
    feature_config = FeatureConfig(
        max_detections=args.max_detections,
        normalize_bbox=True,
        include_arm_state=True,
        include_gripper=True,
        include_rel_to_target=True,
    )

    # ─── Dataset laden ───
    try:
        dataset = TeleopDataset(
            data_dir=args.data,
            feature_config=feature_config,
            seq_len=args.seq_len,
            target_class=args.target,
            only_successful=not args.include_failed,
            skip_idle=not args.include_idle,
        )
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        print(f"    Erst Daten aufnehmen: python teleop_recorder.py")
        return

    if len(dataset) < 10:
        print(f"\n  ✗ Zu wenig Daten ({len(dataset)} Samples)!")
        print(f"    Mindestens 10 Samples nötig. Mehr Episoden aufnehmen.")
        return

    # ─── Train/Val Split ───
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ─── Modell erstellen ───
    input_dim = get_feature_dim(feature_config)
    print(f"\n  Input-Dimension: {input_dim}")

    if args.arch == "mlp":
        model = GraspPolicyMLP(
            input_dim=input_dim,
            hidden_dims=args.hidden,
            num_actions=NUM_ACTIONS,
            dropout=args.dropout,
        )
        print(f"  Architektur: MLP {args.hidden}")

    elif args.arch == "lstm":
        model = GraspPolicyLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_actions=NUM_ACTIONS,
            dropout=args.dropout,
        )
        print(f"  Architektur: LSTM (hidden={args.hidden_dim}, layers={args.num_layers})")

    elif args.arch == "transformer":
        model = GraspPolicyTransformer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=4,
            num_layers=args.num_layers,
            num_actions=NUM_ACTIONS,
            dropout=args.dropout,
            max_seq_len=max(args.seq_len, 50),
        )
        print(f"  Architektur: Transformer (hidden={args.hidden_dim}, layers={args.num_layers})")

    # ─── Class Weights ───
    class_weights = None
    if args.balance_classes:
        trainer_tmp = PolicyTrainer(model, device=args.device, lr=args.lr)
        class_weights = trainer_tmp.compute_class_weights(dataset)
        print(f"  Class Weights: {class_weights.round(2)}")
        del trainer_tmp

    # ─── Trainer erstellen ───
    trainer = PolicyTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
    )

    # ─── Training ───
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=args.output,
        patience=args.patience,
    )

    # ─── Analyse ───
    print_confusion_analysis(model, val_loader, trainer.device)

    # ─── History speichern ───
    history_path = args.output.replace(".pt", "_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training-History gespeichert: {history_path}")

    # ─── Zusammenfassung ───
    print(f"\n{'='*60}")
    print(f"  FERTIG!")
    print(f"  Modell:   {args.output}")
    print(f"  History:  {history_path}")
    print(f"  Nutzung:  python run_policy.py --model {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
