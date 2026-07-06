#!/usr/bin/env python3
"""
LeRobot Policy Inference for RoArm-M2-S
=========================================

Loads a trained ACT/Diffusion/TDMPC model and executes the learned policy
on the real robot arm, similar to how teleop_recorder.py controls the arm.

Usage:
    python3 run_policy.py trained_models/model_final.pt
    python3 run_policy.py trained_models/model_final.pt --episodes 5
    python3 run_policy.py trained_models/model_final.pt --port /dev/ttyUSB0
    python3 run_policy.py trained_models/checkpoint_best.pt --speed-scale 0.5

Controls during execution:
    SPACE   → Start/Stop policy execution
    R       → Reset arm to start position
    Q       → Quit
    +/-     → Adjust execution speed
"""

import argparse
import json
import time
import sys
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch is required! Install: pip install torch")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from roarm_m2s import RoArmM2S

# Import model classes from training script
from train_policy import ACTPolicy, DiffusionPolicy, TDMPCPolicy, load_model


class PolicyRunner:
    """
    Runs a trained LeRobot policy on the RoArm-M2-S.
    
    Handles:
    - Model loading and inference
    - State normalization/denormalization
    - Action chunking with temporal ensemble
    - Real-time arm control at consistent rate
    - Live camera preview (optional)
    - Safety limits and emergency stop
    """

    # Joint limits (same as teleop_recorder.py)
    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    # Execution rate
    CONTROL_HZ = 30  # Match recording FPS

    def __init__(self, model_path: str, port: str = None, camera_index: int = 2,
                 speed_scale: float = 1.0, num_episodes: int = 1,
                 temporal_ensemble: bool = True, headless: bool = False):
        self._model_path = model_path
        self._port = port
        self._camera_index = camera_index
        self._speed_scale = speed_scale
        self._num_episodes = num_episodes
        self._temporal_ensemble = temporal_ensemble
        self._headless = headless

        # Hardware
        self._arm: Optional[RoArmM2S] = None
        self._camera = None

        # Model
        self._model = None
        self._stats = {}
        self._config = {}
        self._device = "cpu"
        self._chunk_size = 10

        # State
        self._running = False
        self._executing = False
        self._current_state = np.array([0.0, 0.0, 90.0, 180.0, 0.0, 1.0], dtype=np.float32)
        self._action_queue: List[np.ndarray] = []
        self._action_queue_lock = threading.Lock()

        # Temporal ensemble buffer
        self._ensemble_buffer: List[np.ndarray] = []
        self._ensemble_weights: Optional[np.ndarray] = None

    # ─── Setup ────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialize hardware and model."""
        print("=" * 60)
        print("  🤖 LeRobot Policy Runner - RoArm-M2-S")
        print("=" * 60)

        # 1. Load model
        print(f"\n  [1/3] Loading model: {self._model_path}")
        try:
            self._model = load_model(self._model_path, device=None)
            self._stats = getattr(self._model, '_stats', {})
            self._config = getattr(self._model, '_config', {})
            self._chunk_size = self._config.get('chunk_size', 10)
            self._device = next(self._model.parameters()).device

            num_params = sum(p.numel() for p in self._model.parameters())
            print(f"        ✓ Policy: {self._config.get('policy', 'unknown').upper()}")
            print(f"        ✓ Parameters: {num_params:,}")
            print(f"        ✓ Chunk size: {self._chunk_size}")
            print(f"        ✓ Device: {self._device}")
        except Exception as e:
            print(f"        ✗ Model load failed: {e}")
            return False

        # 2. Connect arm
        print(f"\n  [2/3] Connecting arm...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            print(f"        ✓ Connected: {self._arm.port}")
        except Exception as e:
            print(f"        ✗ Arm connection failed: {e}")
            return False

        # 3. Camera (optional)
        print(f"\n  [3/3] Camera setup...")
        if not self._headless and HAS_CV2:
            self._camera = cv2.VideoCapture(self._camera_index)
            if not self._camera.isOpened():
                # Try fallback indices
                for idx in [0, 2, 1, 4]:
                    self._camera = cv2.VideoCapture(idx)
                    if self._camera.isOpened():
                        self._camera_index = idx
                        break
            if self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"        ✓ Camera {self._camera_index} ready")
            else:
                print(f"        ⚠ No camera found (running without preview)")
                self._camera = None
        else:
            print(f"        ⚠ Headless mode (no preview)")

        # Setup temporal ensemble weights (exponential decay)
        if self._temporal_ensemble:
            weights = np.exp(-0.01 * np.arange(self._chunk_size))
            self._ensemble_weights = weights / weights.sum()

        # Move arm to start position
        print(f"\n  Moving arm to start position...")
        self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
        time.sleep(2.0)
        self._arm.gripper_open()
        time.sleep(0.5)
        self._current_state = np.array([0.0, 0.0, 90.0, 180.0, 0.0, 1.0], dtype=np.float32)

        print(f"\n  ✓ Setup complete!")
        print(f"\n  Controls:")
        print(f"    SPACE  → Start/Stop policy execution")
        print(f"    R      → Reset arm to start position")
        print(f"    +/-    → Adjust speed (current: {self._speed_scale:.1f}x)")
        print(f"    Q      → Quit")
        print()

        return True

    def _execute_led(self, brightness: int):
        """Send LED command if brightness changed."""
        if not hasattr(self, '_last_led_brightness'):
            self._last_led_brightness = -1

        if brightness != self._last_led_brightness:
            self._arm.set_led(brightness)
            self._last_led_brightness = brightness


    def _extract_led_from_action(self, action: np.ndarray) -> int:
        """Extract LED brightness (0-255) from action vector."""
        if len(action) >= 6:
            led_norm = float(np.clip(action[5], 0.0, 1.0))
            return int(round(led_norm * 255))
        return 255  # Default: full brightness


    # ─── Normalization ────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] using training stats."""
        if "observation.state" in self._stats:
            s = self._stats["observation.state"]
            data_min = np.array(s["min"], dtype=np.float32)
            data_max = np.array(s["max"], dtype=np.float32)

            # Pad if needed
            if len(data_min) < len(state):
                data_min = np.pad(data_min, (0, len(state) - len(data_min)), constant_values=0.0)
                data_max = np.pad(data_max, (0, len(state) - len(data_max)), constant_values=1.0)
            elif len(data_min) > len(state):
                data_min = data_min[:len(state)]
                data_max = data_max[:len(state)]

            range_val = np.maximum(data_max - data_min, 1e-6)
            return 2.0 * (state - data_min) / range_val - 1.0
        return state

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] back to real joint values."""
        if "action" in self._stats:
            s = self._stats["action"]
            data_min = np.array(s["min"], dtype=np.float32)
            data_max = np.array(s["max"], dtype=np.float32)

            # Pad if needed
            if len(data_min) < len(action):
                data_min = np.pad(data_min, (0, len(action) - len(data_min)), constant_values=0.0)
                data_max = np.pad(data_max, (0, len(action) - len(data_max)), constant_values=1.0)
            elif len(data_min) > len(action):
                data_min = data_min[:len(action)]
                data_max = data_max[:len(action)]

            range_val = data_max - data_min
            return (action + 1.0) / 2.0 * range_val + data_min
        return action

    # ─── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Run model inference to get action chunk.
        
        Returns:
            actions: [chunk_size, 5] array of denormalized actions
        """
        # Normalize state
        state_norm = self._normalize_state(state)
        state_tensor = torch.from_numpy(state_norm).unsqueeze(0).to(self._device)

        # Forward pass
        policy_type = self._config.get('policy', 'act')

        if policy_type == 'act':
            # ACT returns actions directly
            actions_norm = self._model(state_tensor)  # [1, chunk_size, action_dim]
        elif policy_type in ('diffusion', 'tdmpc'):
            # These use .sample() for inference
            actions_norm = self._model.sample(state_tensor)  # [1, chunk_size, action_dim]
        else:
            actions_norm = self._model(state_tensor)

        # Convert to numpy
        actions_norm = actions_norm.squeeze(0).cpu().numpy()  # [chunk_size, action_dim]

        # Denormalize each action
        actions = np.array([
            self._denormalize_action(actions_norm[i])
            for i in range(actions_norm.shape[0])
        ])

        return actions

    def _get_next_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get next action using temporal ensemble (if enabled).
        
        Temporal ensemble: predict new chunk, blend with remaining
        actions from previous predictions for smoother execution.
        """
        if not self._temporal_ensemble:
            # Simple: predict new chunk, take first action
            if len(self._action_queue) == 0:
                actions = self._predict_actions(state)
                self._action_queue = list(actions)
            return self._action_queue.pop(0)

        # Temporal ensemble: always predict, blend with buffer
        new_actions = self._predict_actions(state)

        if len(self._ensemble_buffer) == 0:
            # First prediction: use directly
            self._ensemble_buffer = list(new_actions)
        else:
            # Blend new predictions with existing buffer
            # New prediction covers next chunk_size steps
            # Existing buffer has remaining steps from previous predictions
            blended = []
            for i in range(min(len(self._ensemble_buffer), self._chunk_size)):
                # Weighted average: more weight on newer predictions for near-future
                w_new = 0.5  # Can be tuned
                w_old = 1.0 - w_new
                if i < len(self._ensemble_buffer):
                    blended_action = w_old * self._ensemble_buffer[i] + w_new * new_actions[i]
                else:
                    blended_action = new_actions[i]
                blended.append(blended_action)

            # Append remaining new actions
            for i in range(len(self._ensemble_buffer), self._chunk_size):
                blended.append(new_actions[i])

            self._ensemble_buffer = blended

        # Pop first action
        if len(self._ensemble_buffer) > 0:
            action = self._ensemble_buffer.pop(0)
        else:
            action = new_actions[0]

        return action

    # ─── Safety ───────────────────────────────────────────────────────────

    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp action to safe limits (6-dim)."""
        clamped = action.copy()
        clamped[0] = np.clip(clamped[0], self.BASE_MIN, self.BASE_MAX)
        clamped[1] = np.clip(clamped[1], self.SHOULDER_MIN, self.SHOULDER_MAX)
        clamped[2] = np.clip(clamped[2], self.ELBOW_MIN, self.ELBOW_MAX)
        clamped[3] = np.clip(clamped[3], self.HAND_MIN, self.HAND_MAX)
        clamped[4] = np.clip(clamped[4], 0.0, 1.0)   # Gripper
        if len(clamped) >= 6:
            clamped[5] = np.clip(clamped[5], 0.0, 1.0)  # LED normalized
        return clamped


    def _smooth_action(self, action: np.ndarray, prev_state: np.ndarray,
                       max_delta_deg: float = 5.0) -> np.ndarray:
        """
        Smooth action to prevent jerky movements.
        Limits maximum change per step.
        """
        max_delta = max_delta_deg / self._speed_scale
        smoothed = action.copy()

        for i in range(4):  # Joint angles only
            delta = action[i] - prev_state[i]
            if abs(delta) > max_delta:
                smoothed[i] = prev_state[i] + np.sign(delta) * max_delta

        return smoothed

    # ─── Execution ────────────────────────────────────────────────────────

    def _execute_gripper(self, gripper: float):
        """Handle gripper with hysteresis."""
        gripper_threshold = 0.5
        current_gripper = self._current_state[4]

        if gripper > gripper_threshold and current_gripper <= gripper_threshold:
            self._arm.gripper_close()
        elif gripper <= gripper_threshold and current_gripper > gripper_threshold:
            self._arm.gripper_open()


    def _execute_action(self, action: np.ndarray):
        """Send action to the arm (joints + gripper + LED)."""
        base_deg = round(float(action[0]), 1)
        shoulder_deg = round(float(action[1]), 1)
        elbow_deg = round(float(action[2]), 1)
        hand_deg = round(float(action[3]), 1)
        gripper = float(action[4])

        # Joint command
        cmd = {
            "T": 122,
            "b": base_deg,
            "s": shoulder_deg,
            "e": elbow_deg,
            "h": hand_deg,
            "spd": 50,
            "acc": 100
        }
        self._arm._send_nowait(cmd)

        # Gripper control
        self._execute_gripper(gripper)

        # LED control
        led_brightness = self._extract_led_from_action(action)
        self._execute_led(led_brightness)

        # Update current state
        self._current_state = action.copy()

    def _reset_arm(self):
        """Reset arm to start position."""
        print("  → Resetting arm to start position...")
        self._arm.gripper_open()
        time.sleep(0.3)
        self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
        time.sleep(2.0)
        self._current_state = np.array([0.0, 0.0, 90.0, 180.0, 0.0, 1.0], dtype=np.float32)
        self._action_queue = []
        self._ensemble_buffer = []
        print("  ✓ Reset complete")

    # ─── Visualization ────────────────────────────────────────────────────

    def _annotate_frame(self, frame, action: Optional[np.ndarray], step: int,
                        total_steps: int, episode: int, fps: float):
        """Draw status overlay on camera frame."""
        if frame is None:
            return

        h_f, w_f = frame.shape[:2]
        state = self._current_state

        # Semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status text
        status = "▶ EXECUTING" if self._executing else "⏸ PAUSED"
        color = (0, 255, 0) if self._executing else (0, 200, 255)
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Episode/Step info
        info = f"Episode {episode} | Step {step} | Speed: {self._speed_scale:.1f}x | FPS: {fps:.0f}"
        cv2.putText(frame, info, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Joint state (bottom)
        state_str = (f"B:{state[0]:.1f} S:{state[1]:.1f} "
                     f"E:{state[2]:.1f} H:{state[3]:.1f} "
                     f"G:{'C' if state[4] > 0.5 else 'O'}")

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h_f - 35), (w_f, h_f), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, state_str, (10, h_f - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Action visualization (if available)
        if action is not None:
            action_str = (f"→ B:{action[0]:.1f} S:{action[1]:.1f} "
                          f"E:{action[2]:.1f} H:{action[3]:.1f} "
                          f"G:{'C' if action[4] > 0.5 else 'O'}")
            cv2.putText(frame, action_str, (10, h_f - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        # Controls hint
        cv2.putText(frame, "SPACE=Start/Stop  R=Reset  +/-=Speed  Q=Quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # ─── Main Loop ────────────────────────────────────────────────────────

    def run(self):
        """Main execution loop."""
        if not self.setup():
            return

        self._running = True
        window_name = "RoArm Policy Runner"

        control_interval = 1.0 / self.CONTROL_HZ
        fps_time = time.time()
        frame_count = 0
        current_fps = 0.0

        episode = 0
        step = 0
        last_action = None

        print(f"\n  Ready! Press SPACE to start policy execution.\n")

        try:
            while self._running:
                loop_start = time.time()

                # ─── Camera frame ───
                frame = None
                if self._camera:
                    self._camera.grab()
                    ret, frame = self._camera.retrieve()
                    if not ret:
                        frame = None

                # ─── Key handling ───
                key = cv2.waitKey(1) & 0xFF if HAS_CV2 and not self._headless else -1

                if key == ord('q'):
                    self._running = False
                    break
                elif key == ord(' '):
                    if not self._executing:
                        # Start execution
                        self._executing = True
                        episode += 1
                        step = 0
                        self._action_queue = []
                        self._ensemble_buffer = []
                        print(f"\n  ▶ Episode {episode} started!")
                    else:
                        # Stop execution
                        self._executing = False
                        print(f"  ⏸ Episode {episode} paused at step {step}")
                elif key == ord('r'):
                    self._executing = False
                    self._reset_arm()
                    episode = 0
                    step = 0
                elif key == ord('+') or key == ord('='):
                    self._speed_scale = min(3.0, self._speed_scale + 0.1)
                    print(f"  ⚡ Speed: {self._speed_scale:.1f}x")
                elif key == ord('-') or key == ord('_'):
                    self._speed_scale = max(0.1, self._speed_scale - 0.1)
                    print(f"  ⚡ Speed: {self._speed_scale:.1f}x")

                # ─── Policy execution ───
                if self._executing:
                    # Get next action from model
                    action = self._get_next_action(self._current_state)

                    # Safety: clamp to limits
                    action = self._clamp_action(action)

                    # Smoothing: limit max change per step
                    action = self._smooth_action(action, self._current_state,
                                                  max_delta_deg=3.0 * self._speed_scale)

                    # Execute on arm
                    self._execute_action(action)
                    last_action = action
                    step += 1

                    # Check if episode should end (e.g., after N steps)
                    max_steps = self._chunk_size * 20  # ~200 steps max
                    if step >= max_steps:
                        self._executing = False
                        print(f"  ■ Episode {episode} complete ({step} steps)")

                        if episode >= self._num_episodes:
                            print(f"\n  ✓ All {self._num_episodes} episodes complete!")
                            self._running = False

                # ─── Visualization ───
                if frame is not None and not self._headless:
                    self._annotate_frame(frame, last_action, step, 0, episode, current_fps)
                    cv2.imshow(window_name, frame)

                # ─── FPS tracking ───
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_time = time.time()

                # ─── Rate limiting ───
                loop_elapsed = time.time() - loop_start
                sleep_time = max(0.001, control_interval - loop_elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n  [Interrupted]")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown."""
        print("\n  Shutting down...")
        self._executing = False

        if self._arm:
            self._arm.gripper_open()
            time.sleep(0.3)
            self._arm.park()
            time.sleep(1.5)
            self._arm.set_led(0)
            self._arm.disconnect()

        if self._camera:
            self._camera.release()

        if HAS_CV2:
            cv2.destroyAllWindows()

        print("  ✓ Done!")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🤖 Run trained LeRobot policy on RoArm-M2-S",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python3 run_policy.py trained_models/model_final.pt

  # Run 3 episodes at half speed
  python3 run_policy.py trained_models/model_final.pt --episodes 3 --speed-scale 0.5

  # Use specific port, no camera
  python3 run_policy.py trained_models/model_final.pt --port /dev/ttyUSB0 --headless

  # Use best checkpoint
  python3 run_policy.py trained_models/checkpoint_best.pt
        """
    )

    parser.add_argument("model_path", type=str,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if not specified)")
    parser.add_argument("--camera", type=int, default=2,
                        help="Camera index (default: 2)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="Execution speed multiplier (default: 1.0)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable temporal ensemble (use raw predictions)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without camera preview")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    runner = PolicyRunner(
        model_path=args.model_path,
        port=args.port,
        camera_index=args.camera,
        speed_scale=args.speed_scale,
        num_episodes=args.episodes,
        temporal_ensemble=not args.no_ensemble,
        headless=args.headless,
    )
    runner.run()


if __name__ == "__main__":
    main()
