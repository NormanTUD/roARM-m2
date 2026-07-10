"""
lib/policy.py — Policy network that ONLY sees bounding boxes, not raw pixels.

The key insight (Faserbündel / fiber bundle idea):
- YOLO handles the complex visual processing (base manifold)
- The policy network only sees normalized bounding box vectors (fiber)
- Movement patterns are learned invariant of absolute camera position
- Distance estimation normalizes depth → position-independent learning

Input to the policy:
  - VisionState vector: [max_objects * 9] (from lib/vision.py)
  - Arm state: [base, shoulder, elbow, hand, gripper] (5 dims)
  Total input: 5 + 27 = 32 dimensions (with 3 objects)

Output:
  - Action chunk: [chunk_size, 6] (base, shoulder, elbow, hand, gripper, led)

This is DRAMATICALLY simpler than learning from raw 640x480x3 images!
The policy can learn useful behaviors from very few demonstrations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BBoxPolicy(nn.Module):
    """
    Policy network that takes bounding box vectors + arm state as input.
    
    Architecture:
    - Input: [arm_state(5) + vision_vector(max_objects*9)]
    - Transformer decoder with action chunking
    - Output: [chunk_size, action_dim(6)]
    
    Much faster to train than image-based policies because:
    1. Input is ~32 dimensions instead of 640*480*3 = 921,600
    2. YOLO already extracted the relevant features
    3. Normalized coordinates make learning position-invariant
    4. Distance estimation provides depth without stereo vision
    """

    def __init__(self, arm_state_dim: int = 5, max_objects: int = 3,
                 features_per_object: int = 9, action_dim: int = 6,
                 chunk_size: int = 10, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()

        self.arm_state_dim = arm_state_dim
        self.vision_dim = max_objects * features_per_object
        self.input_dim = arm_state_dim + self.vision_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding for action chunk
        self.pos_embed = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, arm_state: torch.Tensor,
                vision_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            arm_state: [B, arm_state_dim] — current joint angles + gripper
            vision_vector: [B, vision_dim] — from VisionState.to_vector()
            
        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = arm_state.shape[0]

        # Concatenate inputs
        x = torch.cat([arm_state, vision_vector], dim=-1)  # [B, input_dim]

        # Encode
        memory = self.input_encoder(x).unsqueeze(1)  # [B, 1, hidden_dim]

        # Decode action chunk
        query = self.pos_embed.expand(B, -1, -1)  # [B, chunk_size, hidden_dim]
        decoded = self.transformer(query, memory)  # [B, chunk_size, hidden_dim]

        # Predict actions
        actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
        return actions

    @torch.no_grad()
    def predict(self, arm_state: np.ndarray,
                vision_vector: np.ndarray) -> np.ndarray:
        """
        Convenience method for inference.
        
        Args:
            arm_state: [5] numpy array
            vision_vector: [max_objects*9] numpy array
            
        Returns:
            actions: [chunk_size, action_dim] numpy array
        """
        device = next(self.parameters()).device

        arm_t = torch.from_numpy(arm_state).float().unsqueeze(0).to(device)
        vis_t = torch.from_numpy(vision_vector).float().unsqueeze(0).to(device)

        actions = self.forward(arm_t, vis_t)
        return actions.squeeze(0).cpu().numpy()


class BBoxPolicyRunner:
    """
    Runs the BBox policy in real-time with the arm and vision pipeline.
    
    This is the inference loop that:
    1. Gets YOLO detections → bounding box vector
    2. Reads arm state
    3. Feeds both to the policy network
    4. Executes predicted actions on the arm
    """

    def __init__(self, model_path: str, arm, vision,
                 class_map: Dict[str, int] = None,
                 max_objects: int = 3,
                 speed_scale: float = 1.0):
        self._arm = arm
        self._vision = vision
        self._class_map = class_map or {}
        self._max_objects = max_objects
        self._speed_scale = speed_scale

        # Load model
        self._model = self._load_model(model_path)
        self._action_queue: List[np.ndarray] = []

    def _load_model(self, path: str) -> BBoxPolicy:
        """Load trained BBoxPolicy."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        config = checkpoint.get('config', {})
        model = BBoxPolicy(
            arm_state_dim=config.get('arm_state_dim', 5),
            max_objects=config.get('max_objects', 3),
            features_per_object=config.get('features_per_object', 9),
            action_dim=config.get('action_dim', 6),
            chunk_size=config.get('chunk_size', 10),
            hidden_dim=config.get('hidden_dim', 128),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 3),
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self._stats = checkpoint.get('stats', {})
        return model

    def step(self) -> np.ndarray:
        """
        Execute one step:
        1. Get vision state (bounding boxes only)
        2. Get arm state
        3. Predict action (or use queued action from chunk)
        4. Execute action
        
        Returns the executed action.
        """
        # If we have queued actions from a previous chunk, use those
        if not self._action_queue:
            # Get fresh prediction
            vision_state = self._vision.get_vision_state(
                max_objects=self._max_objects
            )
            vision_vector = vision_state.to_vector(
                max_objects=self._max_objects,
                class_map=self._class_map
            )

            arm_state = self._get_arm_state_vector()

            # Predict action chunk
            actions = self._model.predict(arm_state, vision_vector)
            self._action_queue = list(actions)

        # Pop next action
        action = self._action_queue.pop(0)

        # Execute
        self._execute_action(action)
        return action

    def _get_arm_state_vector(self) -> np.ndarray:
        """Get current arm state as numpy vector."""
        state = self._arm.state
        return np.array([
            state.base_deg,
            state.shoulder_deg,
            state.elbow_deg,
            state.hand_deg,
            0.0 if state.gripper_open else 1.0,
        ], dtype=np.float32)

    def _execute_action(self, action: np.ndarray):
        """Execute a single action on the arm."""
        # Clamp to safe limits
        action[0] = np.clip(action[0], -90, 90)    # base
        action[1] = np.clip(action[1], -30, 60)    # shoulder
        action[2] = np.clip(action[2], 0, 180)     # elbow
        action[3] = np.clip(action[3], 0, 270)     # hand
        action[4] = np.clip(action[4], 0, 1)       # gripper
        if len(action) > 5:
            action[5] = np.clip(action[5], 0, 1)   # LED

        self._arm.move_joints(
            base=float(action[0]),
            shoulder=float(action[1]),
            elbow=float(action[2]),
            hand=float(action[3]),
        )

        # Gripper
        if action[4] > 0.5:
            self._arm.gripper_close()
        else:
            self._arm.gripper_open()

        # LED
        if len(action) > 5:
            self._arm.set_led(int(action[5] * 255))
