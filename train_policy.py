# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "numpy",
#     "pyarrow",
#     "opencv-python",
#     "rich",
#     "matplotlib",
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
LeRobot Training Script for RoArm-M2-S Recordings
===================================================

Trains a policy model on teleop recordings saved in LeRobot format.

Usage:
    python3 train_lerobot.py recordings/lerobot_dataset/
    python3 train_lerobot.py recordings/lerobot_dataset/ --policy act --epochs 100
    python3 train_lerobot.py recordings/lerobot_dataset/ --policy diffusion --batch-size 8

Models:
    act         - Action Chunking Transformer (fast, small, good for limited data)
    diffusion   - Diffusion Policy (higher quality, slower, needs more VRAM)
    tdmpc       - TD-MPC (model-based RL, experimental)

Designed for small hardware (4-8GB VRAM / CPU-only fallback).
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

class LiveTrainingPlot:
    """Live matplotlib plot showing training loss curve."""

    def __init__(self, total_epochs: int):
        if not HAS_MATPLOTLIB:
            self._enabled = False
            return
        self._enabled = True
        self._total_epochs = total_epochs
        self._losses = []
        self._best_losses = []

        plt.ion()  # Interactive mode
        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 5))
        self._fig.suptitle('🤖 LeRobot ACT Training', fontsize=14, fontweight='bold')
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Loss')
        self._ax.set_xlim(0, total_epochs)
        self._ax.grid(True, alpha=0.3)
        self._line_loss, = self._ax.plot([], [], 'b-', linewidth=1.5, label='Train Loss')
        self._line_best, = self._ax.plot([], [], 'r--', linewidth=1, alpha=0.7, label='Best Loss')
        self._ax.legend(loc='upper right')
        self._fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, epoch: int, loss: float, best_loss: float):
        """Update the plot with new data."""
        if not self._enabled:
            return
        self._losses.append(loss)
        self._best_losses.append(best_loss)

        epochs = list(range(1, len(self._losses) + 1))
        self._line_loss.set_data(epochs, self._losses)
        self._line_best.set_data(epochs, self._best_losses)

        # Auto-scale Y axis
        if self._losses:
            valid_losses = [l for l in self._losses if np.isfinite(l)]
            if valid_losses:
                ymin = min(valid_losses) * 0.9
                ymax = max(valid_losses) * 1.1
                self._ax.set_ylim(max(0, ymin), ymax)

        self._ax.set_xlim(0, max(self._total_epochs, epoch + 1))

        # Add text annotation for current values
        self._ax.set_title(
            f'Epoch {epoch}/{self._total_epochs} | '
            f'Loss: {loss:.6f} | Best: {best_loss:.6f}',
            fontsize=11
        )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, path: str):
        """Save the final plot."""
        if not self._enabled:
            return
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📊 Plot saved to: {path}")

    def close(self):
        if self._enabled:
            plt.ioff()
            plt.close(self._fig)

# ─── Rich Console ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeElapsedColumn,
        TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn,
        TaskProgressColumn
    )
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ─── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.models as tv_models
    from torchvision.models import ResNet18_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ─── Optional: Parquet ────────────────────────────────────────────────────────
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# ─── Optional: Video decoding ────────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class LeRobotArmDataset(Dataset):
    """
    Loads a LeRobot-format dataset recorded by teleop_recorder.py.
    
    Each sample contains:
        - observation.state: [base_deg, shoulder_deg, elbow_deg, hand_deg]
        - observation.gripper: [gripper_position]  (0=open, 1=closed)
        - observation.image: [C, H, W] tensor (if video available)
        - action: [base_deg, shoulder_deg, elbow_deg, hand_deg, gripper]
    
    Supports action chunking (predicting N future actions at once).
    """

    def __init__(self, dataset_dir: Path, chunk_size: int = 10,
                 use_images: bool = True, image_size: Tuple[int, int] = (128, 128)):
        self.dataset_dir = Path(dataset_dir)
        self.chunk_size = chunk_size
        self.use_images = use_images and HAS_CV2
        self.image_size = image_size

        # Load metadata
        self.meta_dir = self.dataset_dir / "meta"
        self.data_dir = self.dataset_dir / "data" / "chunk-000"
        self.video_dir = self.dataset_dir / "videos" / "chunk-000" / "observation.images.top"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.info = self._load_info()
        self.episodes = self._load_episodes()
        self.samples = self._build_samples()

        # Normalization stats
        self.stats = self._load_or_compute_stats()

        # Video cache
        self._video_cache: Dict[int, list] = {}

    def _load_info(self) -> dict:
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        return {}

    def _load_episodes(self) -> List[dict]:
        """Load all episode data from parquet/json files."""
        episodes = []
        
        # Find all episode files
        parquet_files = sorted(self.data_dir.glob("episode_*.parquet"))
        json_files = sorted(self.data_dir.glob("episode_*.json"))

        files = parquet_files if parquet_files else json_files

        for ep_file in files:
            ep_data = self._load_episode_file(ep_file)
            if ep_data:
                episodes.append({
                    "path": ep_file,
                    "frames": ep_data,
                    "num_frames": len(ep_data)
                })

        return episodes

    def _load_episode_file(self, path: Path) -> List[dict]:
        """Load a single episode file."""
        if path.suffix == ".parquet" and HAS_PARQUET:
            table = pq.read_table(path)
            df = table.to_pydict()
            frames = []
            num_rows = len(df.get("frame_index", df.get("index", [])))
            for i in range(num_rows):
                frame = {}
                for key, values in df.items():
                    frame[key] = values[i]
                frames.append(frame)
            return frames
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "frames" in data:
                return data["frames"]
        return []

    def _build_samples(self) -> List[Tuple[int, int]]:
        """
        Build list of (episode_idx, frame_idx) pairs.
        Each sample needs `chunk_size` future frames for action targets.
        """
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            max_start = ep["num_frames"] - self.chunk_size
            for frame_idx in range(max(1, max_start)):
                samples.append((ep_idx, frame_idx))
        return samples

    def _load_or_compute_stats(self) -> dict:
        """Compute normalization statistics from actual data (ignore stale stats.json)."""
        all_states = []
        all_actions = []
        for ep in self.episodes:
            for frame in ep["frames"]:
                state = self._extract_state(frame)
                action = self._extract_action(frame)
                if state is not None:
                    all_states.append(state)
                if action is not None:
                    all_actions.append(action)

        if not all_states or not all_actions:
            # Try loading from file as fallback
            stats_path = self.meta_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    return json.load(f)
            return {}

        states_arr = np.array(all_states)
        actions_arr = np.array(all_actions)
        
        stats = {
            "observation.state": {
                "mean": states_arr.mean(axis=0).tolist(),
                "std": np.maximum(states_arr.std(axis=0), 1e-6).tolist(),
                "min": states_arr.min(axis=0).tolist(),
                "max": states_arr.max(axis=0).tolist(),
            },
            "action": {
                "mean": actions_arr.mean(axis=0).tolist(),
                "std": np.maximum(actions_arr.std(axis=0), 1e-6).tolist(),
                "min": actions_arr.min(axis=0).tolist(),
                "max": actions_arr.max(axis=0).tolist(),
            }
        }
        
        # Print for debugging
        print(f"\n  📊 Computed Stats:")
        print(f"    State range: {states_arr.min(axis=0)} → {states_arr.max(axis=0)}")
        print(f"    Action range: {actions_arr.min(axis=0)} → {actions_arr.max(axis=0)}")
        
        return stats

    def _extract_state(self, frame: dict) -> Optional[np.ndarray]:
        """Extract state vector from frame data (including LED)."""
        if "observation.state" in frame:
            s = frame["observation.state"]
            g = frame.get("observation.gripper", [0.0])
            if isinstance(g, list):
                g = g[0]
            led = frame.get("led_brightness", 255) / 255.0
            return np.array(s + [float(g), led], dtype=np.float32)
        elif "arm_state" in frame:
            arm = frame["arm_state"]
            led = frame.get("led_brightness", 255) / 255.0
            return np.array([
                arm["base_deg"], arm["shoulder_deg"],
                arm["elbow_deg"], arm["hand_deg"],
                0.0 if arm.get("gripper_open", True) else 1.0,
                led
            ], dtype=np.float32)
        return None

    def _extract_action(self, frame: dict) -> Optional[np.ndarray]:
        """Extract action vector from frame data (including LED)."""
        if "action" in frame:
            a = frame["action"]
            if isinstance(a, list) and len(a) >= 6:
                return np.array(a[:6], dtype=np.float32)
            elif isinstance(a, list) and len(a) >= 5:
                # Legacy 5-dim: append LED from frame metadata
                led = frame.get("led_brightness", 255) / 255.0
                return np.array(a[:5] + [led], dtype=np.float32)
        return None

    def _get_image(self, ep_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """Load image frame from video file."""
        if not self.use_images:
            return None

        # Check video cache
        if ep_idx not in self._video_cache:
            video_path = self.video_dir / f"episode_{ep_idx:06d}.mp4"
            if not video_path.exists():
                return None

            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize
                frame = cv2.resize(frame, self.image_size)
                # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
                frames.append(frame)
            cap.release()
            self._video_cache[ep_idx] = frames

        frames = self._video_cache.get(ep_idx, [])
        if frame_idx < len(frames):
            return frames[frame_idx]
        return None

    def _normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Normalize data to [-1, 1] using min-max scaling."""
        if key in self.stats:
            data_min = np.array(self.stats[key]["min"], dtype=np.float32)
            data_max = np.array(self.stats[key]["max"], dtype=np.float32)

            # Pad/trim if needed
            if len(data_min) < len(data):
                # Pad with reasonable defaults for gripper (0-1 range)
                data_min = np.pad(data_min, (0, len(data) - len(data_min)), constant_values=0.0)
                data_max = np.pad(data_max, (0, len(data) - len(data_max)), constant_values=1.0)
            elif len(data_min) > len(data):
                data_min = data_min[:len(data)]
                data_max = data_max[:len(data)]

            # Prevent division by zero
            range_val = data_max - data_min
            range_val = np.maximum(range_val, 1e-6)

            # Scale to [-1, 1]
            return 2.0 * (data - data_min) / range_val - 1.0
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        ep = self.episodes[ep_idx]
        frames = ep["frames"]

        # Current state
        current_frame = frames[frame_idx]
        state = self._extract_state(current_frame)
        if state is None:
            state = np.zeros(5, dtype=np.float32)

        # Action chunk (next chunk_size actions)
        action_chunk = []
        for i in range(self.chunk_size):
            future_idx = min(frame_idx + i, ep["num_frames"] - 1)
            action = self._extract_action(frames[future_idx])
            if action is None:
                action = np.zeros(5, dtype=np.float32)
            action_chunk.append(action)
        action_chunk = np.stack(action_chunk, axis=0)  # [chunk_size, 5]

        # Normalize
        state_norm = self._normalize(state, "observation.state")
        # Normalize each action in chunk
        action_chunk_norm = np.stack([
            self._normalize(a, "action") for a in action_chunk
        ])

        sample = {
            "observation.state": torch.from_numpy(state_norm),
            "action": torch.from_numpy(action_chunk_norm),
        }

        # Image (optional)
        if self.use_images:
            img = self._get_image(ep_idx, frame_idx)
            if img is not None:
                sample["observation.image"] = torch.from_numpy(img)
            else:
                sample["observation.image"] = torch.zeros(3, *self.image_size)

        return sample


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ACTPolicy(nn.Module):
    """
    Action Chunking with Transformers (ACT).
    
    Supports optional pretrained ResNet18 vision backbone.
    Predicts a chunk of future actions given current state (+ optional image).
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 5,
                 chunk_size: int = 10, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 4,
                 use_images: bool = False, image_size: Tuple[int, int] = (128, 128),
                 use_pretrained_vision: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.use_images = use_images
        self.use_pretrained_vision = use_pretrained_vision

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Image encoder (pretrained or lightweight CNN)
        if use_images:
            if use_pretrained_vision and HAS_TORCHVISION:
                self.image_encoder = build_pretrained_image_encoder(
                    hidden_dim=hidden_dim, freeze_backbone=True
                )
            else:
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional encoding for action chunk
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, image=None):
        """
        Args:
            state: [B, state_dim]
            image: [B, 3, H, W] (optional)
        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = state.shape[0]

        # Encode state
        state_feat = self.state_encoder(state)  # [B, hidden_dim]

        # Encode image and fuse
        if self.use_images and image is not None:
            img_feat = self.image_encoder(image)  # [B, hidden_dim]
            memory = self.fusion(torch.cat([state_feat, img_feat], dim=-1))
        else:
            memory = state_feat

        memory = memory.unsqueeze(1)  # [B, 1, hidden_dim]

        # Decode action chunk
        query = self.pos_embed.expand(B, -1, -1)  # [B, chunk_size, hidden_dim]
        decoded = self.transformer_decoder(query, memory)  # [B, chunk_size, hidden_dim]

        # Predict actions
        actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
        return actions

class DiffusionPolicy(nn.Module):
    """
    Simplified Diffusion Policy for action generation.
    
    Uses a conditional U-Net-style architecture to denoise action chunks.
    Optimized for small hardware with fewer diffusion steps.
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 5,
                 chunk_size: int = 10, hidden_dim: int = 256,
                 num_diffusion_steps: int = 20,
                 use_images: bool = False, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_steps = num_diffusion_steps
        self.use_images = use_images

        # Condition encoder (state + optional image)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if use_images:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, hidden_dim),
                nn.ReLU(),
            )
            cond_dim = hidden_dim * 2
        else:
            cond_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Noise prediction network (MLP-based for efficiency)
        input_dim = chunk_size * action_dim + cond_dim + hidden_dim  # noisy_action + cond + time
        self.noise_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

        # Diffusion schedule (cosine)
        self.register_buffer("betas", self._cosine_schedule(num_diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def _cosine_schedule(self, T: int) -> torch.Tensor:
        steps = torch.linspace(0, T, T + 1)
        alpha_bar = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.999)

    def _encode_condition(self, state, image=None):
        state_feat = self.state_encoder(state)
        if self.use_images and image is not None:
            img_feat = self.image_encoder(image)
            return torch.cat([state_feat, img_feat], dim=-1)
        return state_feat

    def forward(self, state, action, image=None):
        """
        Training forward: predict noise added to action.
        
        Args:
            state: [B, state_dim]
            action: [B, chunk_size, action_dim] (ground truth)
            image: [B, 3, H, W] (optional)
        Returns:
            loss: MSE between predicted and actual noise
        """
        B = state.shape[0]
        device = state.device

        # Flatten action
        action_flat = action.reshape(B, -1)  # [B, chunk_size * action_dim]

        # Random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device)
        t_norm = t.float() / self.num_steps

        # Add noise
        alpha_bar = self.alphas_cumprod[t].unsqueeze(-1)  # [B, 1]
        noise = torch.randn_like(action_flat)
        noisy_action = torch.sqrt(alpha_bar) * action_flat + torch.sqrt(1 - alpha_bar) * noise

        # Condition
        cond = self._encode_condition(state, image)
        time_emb = self.time_embed(t_norm.unsqueeze(-1))

        # Predict noise
        net_input = torch.cat([noisy_action, cond, time_emb], dim=-1)
        noise_pred = self.noise_net(net_input)

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, state, image=None):
        """
        Inference: denoise from random noise to action chunk.
        """
        B = state.shape[0]
        device = state.device

        # Start from noise
        x = torch.randn(B, self.chunk_size * self.action_dim, device=device)

        cond = self._encode_condition(state, image)

        # Reverse diffusion
        for t_idx in reversed(range(self.num_steps)):
            t_norm = torch.full((B, 1), t_idx / self.num_steps, device=device)
            time_emb = self.time_embed(t_norm)

            net_input = torch.cat([x, cond, time_emb], dim=-1)
            noise_pred = self.noise_net(net_input)

            # DDPM update
            alpha = 1.0 - self.betas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            alpha_bar_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0)

            # Mean
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (self.betas[t_idx] / torch.sqrt(1 - alpha_bar)) * noise_pred
            )

            # Add noise (except last step)
            if t_idx > 0:
                sigma = torch.sqrt(self.betas[t_idx])
                x = x + sigma * torch.randn_like(x)

        return x.reshape(B, self.chunk_size, self.action_dim)


class TDMPCPolicy(nn.Module):
    """
    Simplified TD-MPC style policy.
    
    Learns a world model + policy jointly.
    Lightweight version for small hardware.
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 5,
                 chunk_size: int = 10, hidden_dim: int = 256,
                 use_images: bool = False, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.use_images = use_images

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        if use_images:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU(),
            )
            enc_dim = hidden_dim * 2
        else:
            enc_dim = hidden_dim

        # Dynamics model (predicts next latent state)
        self.dynamics = nn.Sequential(
            nn.Linear(enc_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim),
            nn.LayerNorm(enc_dim),
        )

        # Policy (predicts action from latent)
        self.policy = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action, image=None):
        """
        Training: predict actions autoregressively using learned dynamics.
        Returns MSE loss on action predictions.
        """
        B = state.shape[0]

        # Encode
        z = self.encoder(state)
        if self.use_images and image is not None:
            img_feat = self.image_encoder(image)
            z = torch.cat([z, img_feat], dim=-1)

        total_loss = 0.0
        for t in range(self.chunk_size):
            # Predict action
            pred_action = self.policy(z)
            target_action = action[:, t, :]
            total_loss += nn.functional.mse_loss(pred_action, target_action)

            # Step dynamics with ground truth action (teacher forcing)
            z = self.dynamics(torch.cat([z, target_action], dim=-1))

        return total_loss / self.chunk_size

    @torch.no_grad()
    def sample(self, state, image=None):
        """Inference: roll out policy autoregressively."""
        B = state.shape[0]

        z = self.encoder(state)
        if self.use_images and image is not None:
            img_feat = self.image_encoder(image)
            z = torch.cat([z, img_feat], dim=-1)

        actions = []
        for _ in range(self.chunk_size):
            a = self.policy(z)
            actions.append(a)
            z = self.dynamics(torch.cat([z, a], dim=-1))

        return torch.stack(actions, dim=1)  # [B, chunk_size, action_dim]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print a detailed model summary similar to Keras model.summary().
    Shows layer names, output shapes, parameter counts, and which layers are frozen.
    """
    if HAS_RICH:
        table = Table(
            title=f"📐 {model_name} Architecture Summary",
            box=box.ROUNDED,
            border_style="bright_magenta",
            show_lines=True,
        )
        table.add_column("Layer (type)", style="cyan", min_width=40)
        table.add_column("Output Shape", style="green", min_width=20)
        table.add_column("Param #", style="yellow", justify="right", min_width=12)
        table.add_column("Trainable", style="white", justify="center", min_width=10)
    else:
        print(f"\n{'='*90}")
        print(f"  📐 {model_name} Architecture Summary")
        print(f"{'='*90}")
        print(f"  {'Layer (type)':<45} {'Output Shape':<20} {'Param #':>12} {'Trainable':>10}")
        print(f"  {'-'*87}")

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, module in model.named_modules():
        # Skip the top-level module itself and container modules
        if name == "":
            continue
        # Only show leaf modules (no children that are also nn.Module with params)
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        if direct_params == 0:
            # Check if it's a meaningful container (Sequential, etc.) - skip those
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.TransformerDecoder,
                                   nn.TransformerDecoderLayer)):
                continue

        # Count parameters for this layer
        layer_params = sum(p.numel() for p in module.parameters(recurse=False))
        layer_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        layer_frozen = layer_params - layer_trainable

        total_params += layer_params
        trainable_params += layer_trainable
        frozen_params += layer_frozen

        # Get output shape estimate (based on layer type)
        shape_str = _estimate_output_shape(module)
        trainable_str = "✓" if layer_trainable > 0 else "✗ (frozen)"

        layer_type = module.__class__.__name__
        display_name = f"{name} ({layer_type})"

        if layer_params > 0:
            if HAS_RICH:
                style = "dim" if layer_trainable == 0 else ""
                table.add_row(
                    display_name,
                    shape_str,
                    f"{layer_params:,}",
                    trainable_str,
                    style=style,
                )
            else:
                print(f"  {display_name:<45} {shape_str:<20} {layer_params:>12,} {trainable_str:>10}")

    # Summary footer
    size_mb = total_params * 4 / (1024 * 1024)
    trainable_mb = trainable_params * 4 / (1024 * 1024)

    if HAS_RICH:
        console.print(table)
        console.print(f"\n  [bold]Total params:[/bold]      {total_params:,} ({size_mb:.2f} MB)")
        console.print(f"  [bold green]Trainable params:[/bold green]  {trainable_params:,} ({trainable_mb:.2f} MB)")
        if frozen_params > 0:
            console.print(f"  [bold red]Frozen params:[/bold red]     {frozen_params:,} ({frozen_params*4/1024/1024:.2f} MB)")
        console.print()
    else:
        print(f"  {'='*87}")
        print(f"  Total params:      {total_params:,} ({size_mb:.2f} MB)")
        print(f"  Trainable params:  {trainable_params:,} ({trainable_mb:.2f} MB)")
        if frozen_params > 0:
            print(f"  Frozen params:     {frozen_params:,} ({frozen_params*4/1024/1024:.2f} MB)")
        print(f"  {'='*87}\n")


def _estimate_output_shape(module: nn.Module) -> str:
    """Estimate output shape string for a layer."""
    if isinstance(module, nn.Linear):
        return f"[*, {module.out_features}]"
    elif isinstance(module, nn.Conv2d):
        return f"[*, {module.out_channels}, H, W]"
    elif isinstance(module, nn.LayerNorm):
        shape = list(module.normalized_shape)
        return f"[*, {', '.join(str(s) for s in shape)}]"
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return f"[*, C, {module.output_size[0]}, {module.output_size[1]}]"
    elif isinstance(module, nn.Flatten):
        return "[*, flattened]"
    elif isinstance(module, (nn.ReLU, nn.SiLU, nn.GELU)):
        return "[same]"
    elif isinstance(module, nn.Parameter):
        return str(list(module.shape))
    elif isinstance(module, nn.MultiheadAttention):
        return f"[*, *, {module.embed_dim}]"
    return "—"

def download_pretrained_backbone(hidden_dim: int = 256, device: str = "cpu") -> Optional[Dict]:
    """
    Download a pretrained ResNet18 backbone from torchvision and adapt it
    for use as a vision encoder or as initialization for MLP layers.

    Returns a dict with pretrained weight tensors that can be partially loaded.
    """
    if not HAS_TORCHVISION:
        print("  ⚠ torchvision not available, skipping pretrained backbone download")
        return None

    print("  📥 Downloading pretrained ResNet18 backbone from PyTorch Hub...")
    try:
        resnet = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("  ✓ ResNet18 (ImageNet pretrained) downloaded successfully")
        return {"resnet18": resnet}
    except Exception as e:
        print(f"  ⚠ Failed to download pretrained model: {e}")
        return None


def build_pretrained_image_encoder(hidden_dim: int = 256, freeze_backbone: bool = True) -> nn.Module:
    """
    Build an image encoder using pretrained ResNet18 features.

    The backbone is frozen by default (only the projection head trains),
    which dramatically speeds up training and reduces overfitting.
    """
    if not HAS_TORCHVISION:
        return None

    print("  🔧 Building pretrained image encoder (ResNet18 → projection)...")

    # Load pretrained ResNet18
    resnet = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Remove the final FC layer and avgpool - we'll use our own
    # ResNet18 feature extractor outputs 512-dim features
    backbone = nn.Sequential(
        resnet.conv1,      # [B, 64, H/2, W/2]
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,    # [B, 64, H/4, W/4]
        resnet.layer1,     # [B, 64, H/4, W/4]
        resnet.layer2,     # [B, 128, H/8, W/8]
        resnet.layer3,     # [B, 256, H/16, W/16]
        resnet.layer4,     # [B, 512, H/32, W/32]
        nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
        nn.Flatten(),      # [B, 512]
    )

    # Freeze backbone if requested
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print("  ❄️  Backbone frozen (only projection head will train)")

    # Projection head (trainable)
    projection = nn.Sequential(
        nn.Linear(512, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
    )

    encoder = nn.Sequential(backbone, projection)

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  ✓ Image encoder: {total:,} params ({trainable:,} trainable, {total-trainable:,} frozen)")

    return encoder


def initialize_transformer_from_pretrained(model: nn.Module, hidden_dim: int = 256):
    """
    Initialize transformer layers with better weight initialization
    inspired by pretrained language model patterns.

    This uses scaled initialization (GPT-2 style) which helps training
    converge faster even without loading actual pretrained weights.
    """
    print("  🎯 Applying pretrained-style initialization to transformer layers...")

    initialized_count = 0
    for name, param in model.named_parameters():
        if "transformer" in name or "decoder" in name:
            if "weight" in name:
                if param.dim() >= 2:
                    # Xavier uniform for attention/FFN weights (like BERT/GPT init)
                    nn.init.xavier_uniform_(param, gain=0.5)
                    initialized_count += 1
            elif "bias" in name:
                nn.init.zeros_(param)
                initialized_count += 1
        elif "action_head" in name and "weight" in name:
            # Smaller init for output head (reduces initial loss)
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)
                initialized_count += 1

    print(f"  ✓ Initialized {initialized_count} parameter tensors with improved init")

class Trainer:
    """
    Training loop with rich progress bars and model saving.
    """

    POLICY_REGISTRY = {
        "act": ACTPolicy,
        "diffusion": DiffusionPolicy,
        "tdmpc": TDMPCPolicy,
    }

    def __init__(self, args):
        self.args = args
        self.device = self._select_device()
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _select_device(self) -> torch.device:
        """Select best available device."""
        if self.args.device:
            return torch.device(self.args.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _print_banner(self):
        if HAS_RICH:
            console.print(Panel(
                "[bold cyan]🤖 LeRobot Training[/bold cyan]\n"
                f"[dim]RoArm-M2-S Policy Training Pipeline[/dim]",
                box=box.DOUBLE,
                border_style="bright_blue",
                padding=(1, 4)
            ))
        else:
            print("=" * 60)
            print("  🤖 LeRobot Training - RoArm-M2-S")
            print("=" * 60)

    def _print_config(self):
        if HAS_RICH:
            table = Table(title="⚙️  Configuration", box=box.ROUNDED, border_style="cyan")
            table.add_column("Parameter", style="bold yellow")
            table.add_column("Value", style="white")

            table.add_row("Dataset", str(self.args.dataset_dir))
            table.add_row("Policy", self.args.policy)
            table.add_row("Device", str(self.device))
            table.add_row("Epochs", str(self.args.epochs))
            table.add_row("Batch Size", str(self.args.batch_size))
            table.add_row("Learning Rate", f"{self.args.lr:.1e}")
            table.add_row("Chunk Size", str(self.args.chunk_size))
            table.add_row("Hidden Dim", str(self.args.hidden_dim))
            table.add_row("Use Images", str(self.args.use_images))
            table.add_row("Output", str(self.args.output))

            table.add_row("Pretrained", str(self.args.pretrained or "None (from scratch)"))
            table.add_row("Pretrained Vision", "✓ ResNet18" if self.args.use_pretrained_vision else "✗")
            table.add_row("Smart Init", "✓" if self.args.smart_init else "✗")

            console.print(table)
        else:
            print(f"\n  Config:")
            print(f"  Dataset:    {self.args.dataset_dir}")
            print(f"  Policy:     {self.args.policy}")
            print(f"  Device:     {self.device}")
            print(f"  Epochs:     {self.args.epochs}")
            print(f"  Batch Size: {self.args.batch_size}")
            print(f"  LR:         {self.args.lr:.1e}")
            print(f"  Chunk Size: {self.args.chunk_size}")
            print(f"  Output:     {self.args.output}")
            print(f"  Pretrained: {self.args.pretrained or 'None (from scratch)'}")
            print(f"  Pretrained Vision: {'ResNet18' if self.args.use_pretrained_vision else 'No'}")
            print(f"  Smart Init: {'Yes' if self.args.smart_init else 'No'}")

    def _print_dataset_info(self):
        if HAS_RICH:
            table = Table(title="📊 Dataset Info", box=box.ROUNDED, border_style="green")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="white")

            table.add_row("Episodes", str(len(self.dataset.episodes)))
            table.add_row("Total Samples", str(len(self.dataset)))
            table.add_row("Chunk Size", str(self.dataset.chunk_size))
            table.add_row("State Dim", "5 (base, shoulder, elbow, hand, gripper)")
            table.add_row("Action Dim", "5 (base, shoulder, elbow, hand, gripper)")

            if self.dataset.info:
                fps = self.dataset.info.get("fps", "?")
                table.add_row("FPS", str(fps))

            has_video = any((self.dataset.video_dir).glob("*.mp4")) if self.dataset.video_dir.exists() else False
            table.add_row("Has Video", "✓" if has_video else "✗")

            console.print(table)
        else:
            print(f"\n  Dataset Info:")
            print(f"    Episodes:      {len(self.dataset.episodes)}")
            print(f"    Total Samples: {len(self.dataset)}")
            print(f"    Chunk Size:    {self.dataset.chunk_size}")

    def _build_model(self) -> nn.Module:
        """Build the selected policy model with optional pretrained components."""
        policy_cls = self.POLICY_REGISTRY[self.args.policy]

        use_images = self.args.use_images
        if use_images:
            has_video = self.dataset.video_dir.exists() and any(self.dataset.video_dir.glob("*.mp4"))
            if not has_video:
                print("  ⚠ No video files found, disabling image input")
                use_images = False

        kwargs = {
            "state_dim": 6,   # base, shoulder, elbow, hand, gripper, led
            "action_dim": 6,  # base, shoulder, elbow, hand, gripper, led
            "chunk_size": self.args.chunk_size,
            "hidden_dim": self.args.hidden_dim,
            "use_images": use_images,
            "image_size": (128, 128),
        }

        if self.args.policy == "act":
            kwargs["num_heads"] = self.args.num_heads
            kwargs["num_layers"] = self.args.num_layers
            kwargs["use_pretrained_vision"] = self.args.use_pretrained_vision and use_images
        elif self.args.policy == "diffusion":
            kwargs["num_diffusion_steps"] = self.args.diffusion_steps

        model = policy_cls(**kwargs)

        # --- Apply smart initialization ---
        if self.args.smart_init:
            initialize_transformer_from_pretrained(model, self.args.hidden_dim)

        # --- Load pretrained checkpoint (transfer learning) ---
        if self.args.pretrained:
            self._load_pretrained_weights(model)

        return model.to(self.device)

    def _load_pretrained_weights(self, model: nn.Module):
        """Load weights from a pretrained checkpoint, matching what we can."""
        pretrained_path = self.args.pretrained
        if not os.path.exists(pretrained_path):
            print(f"  ⚠ Pretrained checkpoint not found: {pretrained_path}")
            return

        print(f"  📥 Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            pretrained_dict = checkpoint["model_state_dict"]
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()

        # Match layers by name and shape
        matched = {}
        mismatched_shape = []
        missing = []

        for key, value in pretrained_dict.items():
            if key in model_dict:
                if value.shape == model_dict[key].shape:
                    matched[key] = value
                else:
                    mismatched_shape.append(
                        f"    {key}: pretrained {list(value.shape)} vs model {list(model_dict[key].shape)}"
                    )
            else:
                missing.append(key)

        # Apply matched weights
        model_dict.update(matched)
        model.load_state_dict(model_dict)

        # Report
        total_layers = len(model_dict)
        loaded_layers = len(matched)
        print(f"  ✓ Loaded {loaded_layers}/{total_layers} layers from pretrained checkpoint")

        if mismatched_shape:
            print(f"  ⚠ {len(mismatched_shape)} layers skipped (shape mismatch):")
            for msg in mismatched_shape[:5]:  # Show first 5
                print(msg)
            if len(mismatched_shape) > 5:
                print(f"    ... and {len(mismatched_shape) - 5} more")

        if missing:
            print(f"  ℹ️  {len(missing)} pretrained layers not in current model (ignored)")

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _print_model_info(self):
        if HAS_RICH:
            num_params = self._count_parameters(self.model)
            size_mb = num_params * 4 / (1024 * 1024)  # float32

            table = Table(title="🧠 Model Info", box=box.ROUNDED, border_style="magenta")
            table.add_column("Property", style="bold")
            table.add_column("Value", style="white")

            table.add_row("Architecture", self.args.policy.upper())
            table.add_row("Parameters", f"{num_params:,}")
            table.add_row("Size (est.)", f"{size_mb:.1f} MB")
            table.add_row("Hidden Dim", str(self.args.hidden_dim))
            table.add_row("Chunk Size", str(self.args.chunk_size))
            table.add_row("Device", str(self.device))

            if self.args.policy == "act":
                table.add_row("Heads", str(self.args.num_heads))
                table.add_row("Layers", str(self.args.num_layers))
            elif self.args.policy == "diffusion":
                table.add_row("Diffusion Steps", str(self.args.diffusion_steps))

            console.print(table)
        else:
            num_params = self._count_parameters(self.model)
            print(f"\n  Model: {self.args.policy.upper()}")
            print(f"    Parameters: {num_params:,}")
            print(f"    Device:     {self.device}")

    def train(self):
        """Main training loop."""
        self._print_banner()

        # ─── Load Dataset ─────────────────────────────────────────────────
        if HAS_RICH:
            console.print("\n[bold]📂 Loading Dataset...[/bold]")
        else:
            print("\n  Loading Dataset...")

        self.dataset = LeRobotArmDataset(
            dataset_dir=self.args.dataset_dir,
            chunk_size=self.args.chunk_size,
            use_images=self.args.use_images,
        )

        if len(self.dataset) == 0:
            if HAS_RICH:
                console.print("[bold red]✗ No training samples found![/bold red]")
                console.print("[dim]  Make sure the dataset has episodes with enough frames.[/dim]")
            else:
                print("  ✗ No training samples found!")
            sys.exit(1)

        self._print_config()
        self._print_dataset_info()

        # ─── Build Model ─────────────────────────────────────────────────
        if HAS_RICH:
            console.print("\n[bold]🧠 Building Model...[/bold]")
        else:
            print("\n  Building Model...")

        self.model = self._build_model()
        self._print_model_info()

        # Print detailed model summary
        print_model_summary(self.model, model_name=f"{self.args.policy.upper()} Policy")

        # ─── DataLoader ───────────────────────────────────────────────────
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=min(2, os.cpu_count() or 1),
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        # ─── Optimizer & Scheduler ────────────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        total_steps = len(self.dataloader) * self.args.epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.args.lr * 0.01
        )

        # ─── Training Loop ────────────────────────────────────────────────
        if HAS_RICH:
            console.print("\n[bold]🚀 Training Started![/bold]\n")
        else:
            print("\n  🚀 Training Started!\n")

        best_loss = float("inf")
        loss_history = []
        start_time = time.time()

        if HAS_RICH:
            self._train_with_rich_progress(best_loss, loss_history, start_time)
        else:
            self._train_simple(best_loss, loss_history, start_time)

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch. Returns (avg_loss, best_batch_loss)."""
        self.model.train()
        epoch_loss = 0.0
        best_batch = float("inf")
        num_batches = 0

        for batch in self.dataloader:
            state = batch["observation.state"].to(self.device)
            action = batch["action"].to(self.device)
            image = batch.get("observation.image")
            if image is not None:
                image = image.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass depends on policy type
            if self.args.policy == "diffusion":
                loss = self.model(state, action, image)
            elif self.args.policy == "tdmpc":
                loss = self.model(state, action, image)
            else:  # ACT
                pred_actions = self.model(state, image)
                loss = nn.functional.mse_loss(pred_actions, action)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            best_batch = min(best_batch, batch_loss)
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        return avg_loss, best_batch

    def _train_with_rich_progress(self, best_loss, loss_history, start_time):
        """Training loop with rich progress bars and time estimates."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            # Epoch-level progress
            epoch_task = progress.add_task(
                "Epochs", total=self.args.epochs
            )

            # Batch-level progress (will be reset each epoch)
            batch_task = progress.add_task(
                "   Batches", total=len(self.dataloader), visible=True
            )

            for epoch in range(self.args.epochs):
                # Reset batch progress for this epoch
                progress.reset(batch_task, total=len(self.dataloader))
                progress.update(batch_task, description=f"   Epoch {epoch+1} batches")

                # --- Train one epoch with per-batch updates ---
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in self.dataloader:
                    state = batch["observation.state"].to(self.device)
                    action = batch["action"].to(self.device)
                    image = batch.get("observation.image")
                    if image is not None:
                        image = image.to(self.device)

                    self.optimizer.zero_grad()

                    if self.args.policy == "diffusion":
                        loss = self.model(state, action, image)
                    elif self.args.policy == "tdmpc":
                        loss = self.model(state, action, image)
                    else:  # ACT
                        pred_actions = self.model(state, image)
                        loss = nn.functional.mse_loss(pred_actions, action)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    # Update batch progress bar
                    progress.update(batch_task, advance=1)

                avg_loss = epoch_loss / max(num_batches, 1)
                loss_history.append(avg_loss)

                # Track best
                improved = ""
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    improved = " â­�"
                    self._save_checkpoint("best")

                # Update epoch progress
                lr = self.optimizer.param_groups[0]["lr"]
                progress.update(
                    epoch_task,
                    advance=1,
                    description=(
                        f"Epoch {epoch+1}/{self.args.epochs}"
                        f"Loss: {avg_loss:.6f}{improved}"
                        f"Best: {best_loss:.6f}"
                        f"LR: {lr:.2e}"
                    )
                )

                # Periodic checkpoint
                if (epoch + 1) % self.args.save_every == 0:
                    self._save_checkpoint(f"epoch_{epoch+1}")

        self._save_final(best_loss, loss_history, start_time)

    def _train_simple(self, best_loss, loss_history, start_time):
        """Training loop without rich (fallback)."""
        # Initialize live plot
        plot = LiveTrainingPlot(self.args.epochs) if HAS_MATPLOTLIB else None

        for epoch in range(self.args.epochs):
            avg_loss, best_batch = self._train_epoch(epoch)
            loss_history.append(avg_loss)

            improved = ""
            if avg_loss < best_loss:
                best_loss = avg_loss
                improved = " *"
                self._save_checkpoint("best")

            # Update live plot
            if plot:
                plot.update(epoch + 1, avg_loss, best_loss)

            # Print every N epochs
            if (epoch + 1) % max(1, self.args.epochs // 20) == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch+1:4d}/{self.args.epochs} │ "
                    f"Loss: {avg_loss:.6f}{improved} │ "
                    f"Best: {best_loss:.6f} │ "
                    f"LR: {lr:.2e} │ "
                    f"{elapsed:.0f}s"
                )

            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

        # Save plot
        if plot:
            output_dir = Path(self.args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot.save(str(output_dir / "training_loss.png"))
            plot.close()

        self._save_final(best_loss, loss_history, start_time)

    def _save_checkpoint(self, name: str):
        """Save a model checkpoint."""
        output_dir = Path(self.args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "use_pretrained_vision": getattr(self.args, "use_pretrained_vision", False),
            "smart_init": getattr(self.args, "smart_init", True),
            "config": {
                "policy": self.args.policy,
                "state_dim": 6,
                "action_dim": 6,
                "chunk_size": self.args.chunk_size,
                "hidden_dim": self.args.hidden_dim,
                "num_heads": getattr(self.args, "num_heads", 4),
                "num_layers": getattr(self.args, "num_layers", 4),
                "diffusion_steps": getattr(self.args, "diffusion_steps", 20),
                "use_images": self.args.use_images,
            },
            "stats": self.dataset.stats if self.dataset else {},
        }

        path = output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)

    def _save_final(self, best_loss, loss_history, start_time):
        """Save final model and print summary."""
        total_time = time.time() - start_time
        output_dir = Path(self.args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final model (inference-ready)
        final_path = output_dir / "model_final.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "policy": self.args.policy,
                "state_dim": 6,
                "action_dim": 6,
                "chunk_size": self.args.chunk_size,
                "hidden_dim": self.args.hidden_dim,
                "num_heads": getattr(self.args, "num_heads", 4),
                "num_layers": getattr(self.args, "num_layers", 4),
                "diffusion_steps": getattr(self.args, "diffusion_steps", 20),
                "use_images": self.args.use_images,
            },
            "stats": self.dataset.stats if self.dataset else {},
        }, final_path)

        # Save loss history
        history_path = output_dir / "loss_history.json"
        with open(history_path, "w") as f:
            json.dump({"losses": loss_history}, f)

        # Print summary
        if HAS_RICH:
            console.print()
            panel_content = (
                f"[bold green]✓ Training Complete![/bold green]\n\n"
                f"  Best Loss:    [cyan]{best_loss:.6f}[/cyan]\n"
                f"  Final Loss:   [cyan]{loss_history[-1]:.6f}[/cyan]\n"
                f"  Total Time:   [cyan]{total_time:.1f}s[/cyan]\n"
                f"  Epochs:       [cyan]{self.args.epochs}[/cyan]\n\n"
                f"  📁 Model saved to:\n"
                f"     [yellow]{final_path}[/yellow]\n"
                f"     [yellow]{output_dir / 'checkpoint_best.pt'}[/yellow]\n\n"
                f"  To use this model for inference:\n"
                f"  [dim]model = load_model('{final_path}')[/dim]"
            )
            console.print(Panel(panel_content, title="🏁 Summary", border_style="green", box=box.DOUBLE))
        else:
            print(f"\n{'='*60}")
            print(f"  ✓ Training Complete!")
            print(f"    Best Loss:  {best_loss:.6f}")
            print(f"    Final Loss: {loss_history[-1]:.6f}")
            print(f"    Time:       {total_time:.1f}s")
            print(f"    Model:      {final_path}")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = None) -> nn.Module:
    """
    Load a trained model for inference.
    
    Usage:
        model = load_model("output/model_final.pt")
        # Get action chunk from current state
        state = torch.tensor([[0.0, 0.0, 90.0, 180.0, 0.0]])
        actions = model.sample(state)  # or model(state) for ACT
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Rebuild model
    policy_map = {
        "act": ACTPolicy,
        "diffusion": DiffusionPolicy,
        "tdmpc": TDMPCPolicy,
    }

    policy_cls = policy_map[config["policy"]]
    kwargs = {
        "state_dim": config["state_dim"],
        "action_dim": config["action_dim"],
        "chunk_size": config["chunk_size"],
        "hidden_dim": config["hidden_dim"],
        "use_images": config.get("use_images", False),
    }

    if config["policy"] == "act":
        kwargs["num_heads"] = config.get("num_heads", 4)
        kwargs["num_layers"] = config.get("num_layers", 4)
    elif config["policy"] == "diffusion":
        kwargs["num_diffusion_steps"] = config.get("diffusion_steps", 20)

    model = policy_cls(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Attach stats for denormalization
    model._stats = checkpoint.get("stats", {})
    model._config = config

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="🤖 LeRobot Training - Train policies on RoArm-M2-S recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ACT policy (recommended for small datasets)
  python3 train_lerobot.py recordings/lerobot_dataset/

  # Train with diffusion policy
  python3 train_lerobot.py recordings/lerobot_dataset/ --policy diffusion

  # Smaller model for Raspberry Pi / low VRAM
  python3 train_lerobot.py recordings/lerobot_dataset/ --hidden-dim 128 --num-layers 2

  # Longer training with custom output
  python3 train_lerobot.py recordings/lerobot_dataset/ --epochs 500 --output models/my_policy/

  # Force CPU training
  python3 train_lerobot.py recordings/lerobot_dataset/ --device cpu
        """
    )

    parser.add_argument(
        "dataset_dir", type=str,
        help="Path to LeRobot dataset directory (e.g., recordings/lerobot_dataset/)"
    )

    # Model selection
    parser.add_argument(
        "--policy", type=str, default="act",
        choices=["act", "diffusion", "tdmpc"],
        help="Policy architecture (default: act)"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (default: 500)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension (default: 256)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size (default: 10)")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer heads (ACT, default: 4)")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers (ACT, default: 4)")
    parser.add_argument("--diffusion-steps", type=int, default=20, help="Diffusion steps (default: 20)")

    # Images
    parser.add_argument("--use-images", action="store_true", help="Use camera images as input")
    parser.add_argument("--no-images", action="store_true", help="Disable image input (default)")

    # Output
    parser.add_argument("--output", type=str, default="trained_models/", help="Output directory for model")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N epochs")

    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu, auto-detected)")

    # Pretrained / Transfer Learning
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained checkpoint for fine-tuning (resumes weights)"
    )
    parser.add_argument(
        "--use-pretrained-vision", action="store_true",
        help="Use pretrained ResNet18 as image encoder backbone (auto-downloads)"
    )
    parser.add_argument(
        "--freeze-vision", action="store_true", default=True,
        help="Freeze pretrained vision backbone (default: True)"
    )
    parser.add_argument(
        "--no-freeze-vision", action="store_true",
        help="Unfreeze pretrained vision backbone (fine-tune everything)"
    )
    parser.add_argument(
        "--smart-init", action="store_true", default=True,
        help="Use improved weight initialization for transformer (default: True)"
    )
    parser.add_argument(
        "--no-smart-init", action="store_true",
        help="Disable improved weight initialization"
    )

    args = parser.parse_args()

    # Handle image flags
    if args.no_images:
        args.use_images = False

    # Handle freeze/init flags
    if args.no_freeze_vision:
        args.freeze_vision = False
    if args.no_smart_init:
        args.smart_init = False

    return args


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_TORCH:
        print("ERROR: PyTorch is required!")
        print("  Install: pip install torch")
        sys.exit(1)

    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited because you pressed CTRL-c")
        sys.exit(0)
