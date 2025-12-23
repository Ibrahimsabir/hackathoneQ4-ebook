# Module 4: Vision–Language–Action (VLA)

## Chapter 4.7: Advanced VLA Topics and Applications

This chapter explores advanced topics in Vision-Language-Action (VLA) systems, focusing on cutting-edge techniques, real-world applications, and emerging trends in multimodal AI for robotics. Advanced VLA systems go beyond basic perception-action loops to enable sophisticated cognitive behaviors, long-horizon planning, and complex human-robot interaction.

### Advanced VLA Architectures

Modern VLA systems employ sophisticated architectures that can handle complex, multi-step tasks requiring both perception and reasoning. Key advanced architectures include:

- **Transformer-based VLA Models**: Large-scale multimodal transformers that process vision, language, and action jointly
- **Hierarchical VLA Systems**: Multi-level architectures with high-level planning and low-level control
- **Memory-Augmented VLA**: Systems with explicit memory for long-term reasoning
- **Embodied VLA Agents**: End-to-end trainable agents that learn from embodied experience

### Advanced VLA System Architecture

```
+-------------------------+
|     High-Level Planner  |
|    (Long-horizon goals) |
+-------------------------+
|     VLA Policy Network  |
|   (Vision-Language-Action) |
+-------------------------+
|     Low-Level Control   |
|     (Motor Primitives)  |
+-------------------------+
|     Memory System       |
|   (Episodic, Semantic)  |
+-------------------------+
|     External Memory     |
|  (Maps, Objects, Skills)|
+-------------------------+
```

### Large-Scale VLA Model Integration

Implementing integration with large-scale VLA models:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from audio_common_msgs.msg import AudioData
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import json

class AdvancedVLANode(Node):
    def __init__(self):
        super().__init__('advanced_vla')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.vla_status_pub = self.create_publisher(
            String,
            '/vla_advanced_status',
            10
        )

        # Initialize advanced VLA components
        self.vla_model = None
        self.vision_encoder = None
        self.language_encoder = None
        self.action_decoder = None
        self.initialize_advanced_vla_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_voice = None
        self.vla_context = {
            'memory': [],
            'episode_history': [],
            'long_term_goals': [],
            'current_task': None
        }

        # Control timer
        self.advanced_vla_timer = self.create_timer(0.1, self.advanced_vla_loop)

        self.get_logger().info('Advanced VLA system initialized')

    def initialize_advanced_vla_components(self):
        """Initialize advanced VLA components (simplified large-scale model)"""
        try:
            # Vision encoder (Vision Transformer)
            class VisionEncoder(nn.Module):
                def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12):
                    super(VisionEncoder, self).__init__()

                    # Patch embedding
                    self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)

                    # Positional encoding
                    self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))  # 224x224 -> 14x14 patches + 1 CLS

                    # Transformer blocks
                    self.blocks = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=embed_dim,
                            nhead=num_heads,
                            batch_first=True
                        ) for _ in range(depth)
                    ])

                    # CLS token
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

                    self.norm = nn.LayerNorm(embed_dim)

                def forward(self, x):
                    B, C, H, W = x.shape

                    # Patch embedding
                    x = self.patch_embed(x)  # [B, embed_dim, grid_h, grid_w]
                    x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

                    # Add CLS token
                    cls_tokens = self.cls_token.expand(B, -1, -1)
                    x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, embed_dim]

                    # Add positional embedding
                    x = x + self.pos_embed

                    # Apply transformer blocks
                    for block in self.blocks:
                        x = block(x)

                    # Normalize and return CLS token features
                    x = self.norm(x)
                    return x[:, 0]  # Return CLS token features [B, embed_dim]

            # Language encoder (BERT-based)
            class LanguageEncoder(nn.Module):
                def __init__(self, vocab_size=30522, embed_dim=768, max_length=512):
                    super(LanguageEncoder, self).__init__()

                    self.embeddings = nn.Embedding(vocab_size, embed_dim)
                    self.position_embeddings = nn.Embedding(max_length, embed_dim)

                    # Transformer layers
                    self.transformer_blocks = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=embed_dim,
                            nhead=12,
                            batch_first=True
                        ) for _ in range(12)
                    ])

                    self.norm = nn.LayerNorm(embed_dim)

                def forward(self, input_ids, attention_mask=None):
                    B, seq_len = input_ids.shape

                    # Embeddings
                    token_embeddings = self.embeddings(input_ids)
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                    position_embeddings = self.position_embeddings(position_ids)

                    x = token_embeddings + position_embeddings

                    # Apply attention mask if provided
                    if attention_mask is not None:
                        # Create attention mask for transformer
                        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                    else:
                        extended_attention_mask = None

                    # Apply transformer blocks
                    for block in self.transformer_blocks:
                        x = block(x)

                    # Pool representations (mean pooling)
                    if attention_mask is not None:
                        # Mask out padding tokens before pooling
                        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
                        x = x * mask_expanded
                        pooled = torch.sum(x, 1) / torch.sum(mask_expanded, 1)
                    else:
                        pooled = torch.mean(x, dim=1)

                    x = self.norm(pooled)
                    return x

            # VLA Fusion Network
            class VLAFusionNetwork(nn.Module):
                def __init__(self, vision_dim=768, language_dim=768, action_dim=2):
                    super(VLAFusionNetwork, self).__init__()

                    # Cross-modal attention
                    self.vision_language_attention = nn.MultiheadAttention(
                        embed_dim=768,
                        num_heads=12,
                        batch_first=True
                    )

                    # Memory-augmented processing
                    self.memory_processor = nn.LSTM(
                        input_size=768,
                        hidden_size=512,
                        num_layers=2,
                        batch_first=True
                    )

                    # Action decoder
                    self.action_decoder = nn.Sequential(
                        nn.Linear(768 + 512, 512),  # Vision+Language + Memory
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_dim)  # [linear_x, angular_z]
                    )

                def forward(self, vision_features, language_features, memory_state=None):
                    # Cross-modal attention between vision and language
                    vision_lang, _ = self.vision_language_attention(
                        query=language_features.unsqueeze(1),  # [B, 1, 768]
                        key=vision_features.unsqueeze(1),      # [B, 1, 768]
                        value=vision_features.unsqueeze(1)     # [B, 1, 768]
                    )

                    # Combine features
                    combined_features = vision_lang.squeeze(1) + language_features  # [B, 768]

                    # Process with memory (if available)
                    if memory_state is not None:
                        memory_features, (h_n, c_n) = self.memory_processor(
                            combined_features.unsqueeze(1),  # [B, 1, 768]
                            memory_state
                        )
                        memory_features = memory_features.squeeze(1)  # [B, 512]

                        # Combine with current features
                        final_features = torch.cat([combined_features, memory_features], dim=1)
                    else:
                        # Use only current features
                        final_features = combined_features

                    # Decode action
                    action_output = self.action_decoder(final_features)

                    return action_output, (h_n, c_n) if memory_state is not None else None

            # Initialize components
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
            self.vla_fusion = VLAFusionNetwork()

            # Set to evaluation mode
            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.vla_fusion.eval()

            self.get_logger().info('Advanced VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced VLA components: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_voice = msg.data

    def advanced_vla_loop(self):
        """Main advanced VLA processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_voice is None):
            return

        try:
            # Extract features from all modalities
            vision_features = self.extract_vision_features_advanced(self.current_image)
            language_features = self.extract_language_features_advanced(self.current_voice)

            if vision_features is not None and language_features is not None:
                # Perform VLA fusion with memory
                action_output, new_memory_state = self.perform_advanced_vla_fusion(
                    vision_features, language_features, self.vla_context.get('memory_state', None)
                )

                # Convert to action command
                cmd = self.convert_vla_output_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Update VLA context
                self.vla_context['memory_state'] = new_memory_state
                self.vla_context['episode_history'].append({
                    'timestamp': time.time(),
                    'command': self.current_voice,
                    'vision_features': vision_features.detach().cpu().numpy(),
                    'action': [float(cmd.linear.x), float(cmd.angular.z)] if cmd else [0.0, 0.0],
                    'confidence': float(torch.max(torch.abs(action_output)).item())
                })

                # Publish VLA status
                vla_status_msg = String()
                vla_status_msg.data = json.dumps({
                    'command': self.current_voice,
                    'vision_features_shape': vision_features.shape if vision_features is not None else None,
                    'language_features_shape': language_features.shape if language_features is not None else None,
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    },
                    'confidence': float(torch.max(torch.abs(action_output)).item()) if action_output is not None else 0.0,
                    'episode_length': len(self.vla_context['episode_history'])
                })
                self.vla_status_pub.publish(vla_status_msg)

                self.get_logger().info(
                    f'Advanced VLA - Command: "{self.current_voice}", '
                    f'Vision: {vision_features.shape if vision_features is not None else "None"}, '
                    f'Language: {language_features.shape if language_features is not None else "None"}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}, '
                    f'Confidence: {float(torch.max(torch.abs(action_output)).item()):.3f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in advanced VLA processing: {str(e)}')

    def extract_vision_features_advanced(self, image_msg):
        """Extract advanced vision features using ViT encoder"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Preprocess image
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
            image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features = self.vision_encoder(image_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting advanced vision features: {str(e)}')
            return None

    def extract_language_features_advanced(self, command_text):
        """Extract advanced language features using BERT-style encoder"""
        try:
            # Simple tokenization (in real implementation, use proper tokenizer)
            tokens = command_text.lower().split()
            token_ids = [hash(token) % 30000 for token in tokens]  # Simulate BERT vocab

            # Add special tokens
            token_ids = [101] + token_ids + [102]  # [CLS] + tokens + [SEP]

            # Pad or truncate
            max_length = 32
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            with torch.no_grad():
                features = self.language_encoder(token_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting advanced language features: {str(e)}')
            return None

    def perform_advanced_vla_fusion(self, vision_features, language_features, memory_state):
        """Perform advanced VLA fusion with memory"""
        try:
            with torch.no_grad():
                action_output, new_memory_state = self.vla_fusion(
                    vision_features, language_features, memory_state
                )
                return action_output, new_memory_state

        except Exception as e:
            self.get_logger().error(f'Error in VLA fusion: {str(e)}')
            return None, memory_state

    def convert_vla_output_to_command(self, action_output):
        """Convert VLA output to robot command"""
        if action_output is None:
            return None

        try:
            action_values = action_output[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting VLA output to command: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    advanced_vla = AdvancedVLANode()

    try:
        rclpy.spin(advanced_vla)
    except KeyboardInterrupt:
        advanced_vla.get_logger().info('Advanced VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_vla.action_pub.publish(cmd)

        advanced_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Hierarchical VLA Systems

Implementing hierarchical VLA systems for long-horizon tasks:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Path
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json

class HierarchicalVLANode(Node):
    def __init__(self):
        super().__init__('hierarchical_vla')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            10
        )

        self.high_level_cmd_sub = self.create_subscription(
            String,
            '/high_level_command',
            self.high_level_command_callback,
            10
        )

        self.low_level_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.high_level_status_pub = self.create_publisher(
            String,
            '/hierarchical_vla_status',
            10
        )

        self.subgoal_pub = self.create_publisher(
            PoseStamped,
            '/subgoal_pose',
            10
        )

        # Initialize hierarchical VLA components
        self.high_level_planner = None
        self.low_level_controller = None
        self.task_decomposer = None
        self.initialize_hierarchical_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.high_level_command = None
        self.current_plan = []
        self.current_subgoal = None
        self.plan_step = 0

        # Hierarchical state
        self.hierarchy_state = {
            'current_task': None,
            'task_progress': 0.0,
            'subgoals_completed': 0,
            'total_subgoals': 0,
            'hierarchical_confidence': 0.0
        }

        # Control timers
        self.high_level_timer = self.create_timer(1.0, self.high_level_planning_loop)  # 1Hz
        self.low_level_timer = self.create_timer(0.05, self.low_level_control_loop)    # 20Hz

        self.get_logger().info('Hierarchical VLA system initialized')

    def initialize_hierarchical_components(self):
        """Initialize hierarchical VLA components"""
        try:
            # High-level planner (task decomposition)
            class HighLevelPlanner(nn.Module):
                def __init__(self):
                    super(HighLevelPlanner, self).__init__()

                    # Vision processing for scene understanding
                    self.scene_encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 7, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 5, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )

                    # Language processing for command understanding
                    self.command_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # Task decomposition network
                    self.task_decomposer = nn.Sequential(
                        nn.Linear(256 + 256, 512),  # Vision + Language
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),  # Number of subtasks
                        nn.Softmax(dim=1)
                    )

                    # Subgoal generator
                    self.subgoal_generator = nn.Sequential(
                        nn.Linear(256 + 256, 256),  # Vision + Language
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),  # [x, y, theta] * 2 (start + goal)
                        nn.Tanh()  # Normalize to [-1, 1] range
                    )

                def forward(self, vision_input, language_input):
                    # Process vision input
                    vision_features = self.scene_encoder(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)

                    # Process language input
                    lang_embedded = self.command_encoder[0](language_input)
                    lang_lstm_out, _ = self.command_encoder[1](lang_embedded)
                    lang_features = self.command_encoder[2](lang_lstm_out[:, -1, :])  # Use last output
                    lang_features = F.normalize(lang_features, dim=1)

                    # Decompose task
                    task_probs = self.task_decomposer(
                        torch.cat([vision_features, lang_features], dim=1)
                    )

                    # Generate subgoals
                    subgoal_output = self.subgoal_generator(
                        torch.cat([vision_features, lang_features], dim=1)
                    )

                    return task_probs, subgoal_output

            # Low-level controller (execution)
            class LowLevelController(nn.Module):
                def __init__(self):
                    super(LowLevelController, self).__init__()

                    # Sensor fusion for current state
                    self.state_encoder = nn.Sequential(
                        nn.Linear(360 + 6, 256),  # LiDAR + pose (x, y, theta, vx, vy, omega)
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64)
                    )

                    # Goal encoder
                    self.goal_encoder = nn.Sequential(
                        nn.Linear(3, 64),  # [x, y, theta] goal
                        nn.ReLU(),
                        nn.Linear(64, 64)
                    )

                    # Action decoder
                    self.action_decoder = nn.Sequential(
                        nn.Linear(64 + 64, 128),  # State + Goal
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z]
                    )

                def forward(self, state_input, goal_input):
                    state_features = self.state_encoder(state_input)
                    goal_features = self.goal_encoder(goal_input)

                    action_input = torch.cat([state_features, goal_features], dim=1)
                    action_output = self.action_decoder(action_input)

                    return action_output

            # Task decomposer
            class TaskDecomposer:
                def __init__(self, node):
                    self.node = node

                def decompose_task(self, command, scene_analysis):
                    """Decompose high-level command into subtasks"""
                    subtasks = []

                    command_lower = command.lower()

                    if 'go to' in command_lower or 'move to' in command_lower:
                        # Navigation task - decompose into path following
                        subtasks = [
                            {'type': 'navigate_to_location', 'description': 'Navigate to specified location'},
                            {'type': 'avoid_obstacles', 'description': 'Avoid obstacles during navigation'},
                            {'type': 'reach_destination', 'description': 'Arrive at destination'}
                        ]
                    elif 'pick up' in command_lower or 'grasp' in command_lower:
                        # Manipulation task - decompose into approach, grasp, lift
                        subtasks = [
                            {'type': 'approach_object', 'description': 'Approach the target object'},
                            {'type': 'identify_grasp_point', 'description': 'Identify suitable grasp point'},
                            {'type': 'execute_grasp', 'description': 'Execute grasping motion'},
                            {'type': 'lift_object', 'description': 'Lift object to safe height'}
                        ]
                    elif 'follow' in command_lower:
                        # Following task - decompose into tracking and following
                        subtasks = [
                            {'type': 'detect_target', 'description': 'Detect and identify following target'},
                            {'type': 'maintain_tracking', 'description': 'Maintain tracking of target'},
                            {'type': 'follow_at_distance', 'description': 'Follow target at safe distance'}
                        ]
                    else:
                        # Default navigation task
                        subtasks = [
                            {'type': 'move_forward', 'description': 'Move forward'},
                            {'type': 'avoid_obstacles', 'description': 'Avoid obstacles'},
                            {'type': 'adjust_path', 'description': 'Adjust path as needed'}
                        ]

                    return subtasks

            # Initialize components
            self.high_level_planner = HighLevelPlanner()
            self.low_level_controller = LowLevelController()
            self.task_decomposer = TaskDecomposer(self)

            # Set to evaluation mode
            self.high_level_planner.eval()
            self.low_level_controller.eval()

            self.get_logger().info('Hierarchical VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize hierarchical components: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def high_level_command_callback(self, msg):
        """Process high-level command"""
        self.high_level_command = msg.data

    def high_level_planning_loop(self):
        """High-level planning loop (1Hz)"""
        if (self.high_level_planner is None or
            self.current_image is None or
            self.high_level_command is None):
            return

        try:
            # Extract features
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.tokenize_command(self.high_level_command)

            if vision_features is not None and language_features is not None:
                # Plan at high level
                task_probs, subgoal_output = self.high_level_planner(
                    vision_features, language_features
                )

                # Decompose task into subtasks
                subtasks = self.task_decomposer.decompose_task(
                    self.high_level_command, self.current_detections
                )

                # Generate subgoals
                subgoals = self.generate_subgoals(subgoal_output, subtasks)

                # Update plan
                self.current_plan = subtasks
                self.current_subgoals = subgoals
                self.plan_step = 0

                # Update hierarchy state
                self.hierarchy_state['current_task'] = self.high_level_command
                self.hierarchy_state['total_subgoals'] = len(subtasks)
                self.hierarchy_state['subgoals_completed'] = 0
                self.hierarchy_state['task_progress'] = 0.0

                # Publish first subgoal
                if subgoals:
                    self.publish_subgoal(subgoals[0])

                # Publish status
                status_msg = String()
                status_msg.data = json.dumps({
                    'high_level_command': self.high_level_command,
                    'subtasks': [task['type'] for task in subtasks],
                    'subgoals': len(subgoals),
                    'task_progress': 0.0
                })
                self.high_level_status_pub.publish(status_msg)

                self.get_logger().info(
                    f'High-Level Plan - Command: "{self.high_level_command}", '
                    f'Subtasks: {len(subtasks)}, Subgoals: {len(subgoals)}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in high-level planning: {str(e)}')

    def low_level_control_loop(self):
        """Low-level control loop (20Hz)"""
        if (self.low_level_controller is None or
            self.current_scan is None or
            self.current_subgoals is None or
            self.plan_step >= len(self.current_subgoals)):
            return

        try:
            # Get current subgoal
            current_subgoal = self.current_subgoals[self.plan_step]

            # Get current state (simplified - in real implementation, use odometry)
            current_state = self.get_current_robot_state()

            # Process with low-level controller
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            goal_tensor = torch.FloatTensor(current_subgoal).unsqueeze(0)

            with torch.no_grad():
                action_output = self.low_level_controller(state_tensor, goal_tensor)

            # Convert to command
            cmd = self.convert_action_to_command(action_output)

            if cmd is not None:
                self.low_level_cmd_pub.publish(cmd)

            # Check if subgoal is achieved
            if self.is_subgoal_achieved(current_state, current_subgoal):
                self.plan_step += 1
                self.hierarchy_state['subgoals_completed'] += 1
                self.hierarchy_state['task_progress'] = (
                    self.hierarchy_state['subgoals_completed'] /
                    max(1, self.hierarchy_state['total_subgoals'])
                )

                # Publish next subgoal if available
                if self.plan_step < len(self.current_subgoals):
                    self.publish_subgoal(self.current_subgoals[self.plan_step])

                self.get_logger().info(
                    f'Subgoal completed - Progress: {self.hierarchy_state["task_progress"]:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in low-level control: {str(e)}')

    def extract_vision_features(self, image_msg):
        """Extract vision features for high-level planning"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            return image_tensor

        except Exception as e:
            self.get_logger().error(f'Error extracting vision features: {str(e)}')
            return None

    def tokenize_command(self, command):
        """Tokenize command for language processing"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad to fixed length
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])
            return token_tensor

        except Exception as e:
            self.get_logger().error(f'Error tokenizing command: {str(e)}')
            return None

    def generate_subgoals(self, subgoal_output, subtasks):
        """Generate subgoals from neural network output"""
        try:
            subgoal_values = subgoal_output[0].numpy()

            # Reshape to get [x, y, theta] coordinates for each subgoal
            # Assuming 64 output dimensions for 2 goals (start + end) * 3 coords each = 6
            # For this example, we'll create simple subgoals based on the output
            subgoals = []

            for i, subtask in enumerate(subtasks):
                # Create subgoal based on neural output and task type
                subgoal = {
                    'x': float(subgoal_values[i * 3]) if i * 3 < len(subgoal_values) else 0.0,
                    'y': float(subgoal_values[i * 3 + 1]) if i * 3 + 1 < len(subgoal_values) else 0.0,
                    'theta': float(subgoal_values[i * 3 + 2]) if i * 3 + 2 < len(subgoal_values) else 0.0,
                    'task_type': subtask['type'],
                    'task_description': subtask['description']
                }

                # Scale coordinates appropriately (neural output is normalized to [-1, 1])
                subgoal['x'] = subgoal['x'] * 5.0  # Scale to reasonable range
                subgoal['y'] = subgoal['y'] * 5.0

                subgoals.append(subgoal)

            return subgoals

        except Exception as e:
            self.get_logger().error(f'Error generating subgoals: {str(e)}')
            return []

    def get_current_robot_state(self):
        """Get current robot state (simplified - in real implementation, use odometry)"""
        # In a real implementation, this would get actual robot pose and velocity
        # For this example, we'll simulate state based on current scan and previous commands

        # Simulate position based on scan (simplified)
        if self.current_scan:
            # Calculate some metrics from scan
            ranges = np.array(self.current_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]
            if len(valid_ranges) > 0:
                avg_range = np.mean(valid_ranges)
            else:
                avg_range = 1.0
        else:
            avg_range = 1.0

        # Simulate velocity based on recent commands (simplified)
        simulated_state = [
            avg_range * 0.1,  # x position (simplified)
            0.0,              # y position
            0.0,              # theta orientation
            0.3,              # linear velocity (simplified)
            0.0,              # angular velocity
            0.0               # angular velocity (alternative representation)
        ]

        # Extend with LiDAR data (simplified - take first 354 points to make 360 total)
        if self.current_scan:
            scan_data = np.array(self.current_scan.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)
            scan_data = scan_data[:354] if len(scan_data) >= 354 else np.pad(
                scan_data, (0, 354 - len(scan_data)), mode='constant', constant_values=3.0
            )
            simulated_state.extend(scan_data.tolist())

        return simulated_state

    def convert_action_to_command(self, action_output):
        """Convert neural network output to robot command"""
        if action_output is None:
            return None

        try:
            action_values = action_output[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting action to command: {str(e)}')
            return None

    def is_subgoal_achieved(self, current_state, subgoal):
        """Check if subgoal has been achieved"""
        try:
            # Simplified subgoal achievement check
            # In real implementation, this would be more sophisticated

            current_x = current_state[0]  # From our simulated state
            current_y = current_state[1]

            goal_x = subgoal['x']
            goal_y = subgoal['y']

            distance_to_goal = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)

            # Threshold for goal achievement (simplified)
            threshold = 0.3  # meters

            return distance_to_goal <= threshold

        except Exception as e:
            self.get_logger().error(f'Error checking subgoal achievement: {str(e)}')
            return False

    def publish_subgoal(self, subgoal):
        """Publish subgoal as PoseStamped"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'

            pose_msg.pose.position.x = subgoal['x']
            pose_msg.pose.position.y = subgoal['y']
            pose_msg.pose.position.z = 0.0

            # Convert theta to quaternion
            theta = subgoal['theta']
            pose_msg.pose.orientation.z = np.sin(theta / 2.0)
            pose_msg.pose.orientation.w = np.cos(theta / 2.0)

            self.subgoal_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing subgoal: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    hierarchical_vla = HierarchicalVLANode()

    try:
        rclpy.spin(hierarchical_vla)
    except KeyboardInterrupt:
        hierarchical_vla.get_logger().info('Hierarchical VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        hierarchical_vla.low_level_cmd_pub.publish(cmd)

        hierarchical_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Memory-Augmented VLA Systems

Implementing VLA systems with explicit memory for long-term reasoning:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose, PoseStamped
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import pickle
from datetime import datetime, timedelta

class MemoryAugmentedVLANode(Node):
    def __init__(self):
        super().__init__('memory_augmented_vla')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.memory_status_pub = self.create_publisher(
            String,
            '/memory_vla_status',
            10
        )

        # Initialize memory-augmented VLA components
        self.vision_encoder = None
        self.language_encoder = None
        self.memory_network = None
        self.vla_policy = None
        self.initialize_memory_components()

        # Memory systems
        self.episodic_memory = []  # Recent experiences
        self.semantic_memory = {}  # General knowledge
        self.working_memory = {}   # Current task context
        self.skill_memory = {}     # Learned behaviors

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_memory_state = None

        # Memory management parameters
        self.max_episodic_memory = 1000
        self.episodic_decay_rate = 0.95
        self.semantic_update_threshold = 0.7

        # Control timer
        self.memory_timer = self.create_timer(0.1, self.memory_augmented_vla_loop)

        self.get_logger().info('Memory-augmented VLA system initialized')

    def initialize_memory_components(self):
        """Initialize memory-augmented VLA components"""
        try:
            # Vision encoder with memory attention
            class MemoryAugmentedVisionEncoder(nn.Module):
                def __init__(self):
                    super(MemoryAugmentedVisionEncoder, self).__init__()

                    # Feature extraction
                    self.feature_extractor = nn.Sequential(
                        nn.Conv2d(3, 32, 7, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 5, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((8, 8))
                    )

                    # Memory attention mechanism
                    self.memory_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    # Output projection
                    self.output_proj = nn.Linear(256, 256)

                def forward(self, vision_input, memory_features):
                    # Extract features from vision input
                    features = self.feature_extractor(vision_input)
                    features = features.view(features.size(0), features.size(1), -1)  # [B, C, H*W]
                    features = torch.mean(features, dim=2)  # [B, C]

                    # Apply memory attention
                    if memory_features is not None and memory_features.size(0) > 0:
                        # Reshape for attention
                        features_expanded = features.unsqueeze(1)  # [B, 1, C]
                        memory_expanded = memory_features.unsqueeze(0)  # [1, M, C]

                        # Apply attention
                        attended_features, attention_weights = self.memory_attention(
                            query=features_expanded,
                            key=memory_expanded,
                            value=memory_expanded
                        )

                        # Combine original features with memory-attended features
                        combined_features = features + attended_features.squeeze(1)
                    else:
                        combined_features = features

                    output = self.output_proj(combined_features)
                    return output

            # Language encoder with memory
            class MemoryAugmentedLanguageEncoder(nn.Module):
                def __init__(self):
                    super(MemoryAugmentedLanguageEncoder, self).__init__()

                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True)

                    # Memory attention for language
                    self.memory_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    self.output_proj = nn.Linear(256, 256)

                def forward(self, language_input, memory_features):
                    # Embed and process language
                    embedded = self.embedding(language_input)
                    lstm_out, _ = self.lstm(embedded)
                    features = lstm_out[:, -1, :]  # Use last output

                    # Apply memory attention if memory features available
                    if memory_features is not None and memory_features.size(0) > 0:
                        features_expanded = features.unsqueeze(1)  # [B, 1, 256]
                        memory_expanded = memory_features.unsqueeze(0)  # [1, M, 256]

                        attended_features, _ = self.memory_attention(
                            query=features_expanded,
                            key=memory_expanded,
                            value=memory_expanded
                        )

                        combined_features = features + attended_features.squeeze(1)
                    else:
                        combined_features = features

                    output = self.output_proj(combined_features)
                    return output

            # Memory network
            class MemoryNetwork(nn.Module):
                def __init__(self, memory_dim=256):
                    super(MemoryNetwork, self).__init__()
                    self.memory_dim = memory_dim

                    # Memory writing network
                    self.write_network = nn.Sequential(
                        nn.Linear(256 * 2, 256),  # Vision + Language features
                        nn.ReLU(),
                        nn.Linear(256, memory_dim)
                    )

                    # Memory reading network
                    self.read_network = nn.Sequential(
                        nn.Linear(256 + memory_dim, 256),  # Current + Memory
                        nn.ReLU(),
                        nn.Linear(256, memory_dim)
                    )

                def forward(self, current_features, memory_features, write_flag=True):
                    if write_flag:
                        # Update memory with current experience
                        memory_input = torch.cat([current_features, memory_features], dim=1) if memory_features is not None else current_features
                        new_memory = self.write_network(memory_input)
                        return new_memory
                    else:
                        # Read from memory
                        if memory_features is not None:
                            read_input = torch.cat([current_features, memory_features], dim=1)
                            memory_output = self.read_network(read_input)
                            return memory_output
                        else:
                            return torch.zeros(current_features.size(0), self.memory_dim)

            # VLA policy with memory
            class MemoryAugmentedVLAPolicy(nn.Module):
                def __init__(self):
                    super(MemoryAugmentedVLAPolicy, self).__init__()

                    # Fusion network
                    self.fusion = nn.Sequential(
                        nn.Linear(256 * 3, 512),  # Vision + Language + Memory
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                def forward(self, vision_features, language_features, memory_features):
                    combined_input = torch.cat([
                        vision_features,
                        language_features,
                        memory_features
                    ], dim=1)

                    action_output = self.fusion(combined_input)
                    return action_output

            # Initialize models
            self.vision_encoder = MemoryAugmentedVisionEncoder()
            self.language_encoder = MemoryAugmentedLanguageEncoder()
            self.memory_network = MemoryNetwork()
            self.vla_policy = MemoryAugmentedVLAPolicy()

            # Set to evaluation mode
            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.memory_network.eval()
            self.vla_policy.eval()

            self.get_logger().info('Memory-augmented VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize memory components: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def memory_augmented_vla_loop(self):
        """Main memory-augmented VLA processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Extract features with memory augmentation
            vision_features = self.extract_memory_augmented_vision_features(
                self.current_image, self.get_memory_features()
            )
            language_features = self.extract_memory_augmented_language_features(
                self.current_command, self.get_memory_features()
            )

            if vision_features is not None and language_features is not None:
                # Get current memory state
                current_memory = self.get_memory_features()

                # Perform VLA with memory
                action_output = self.perform_memory_augmented_vla(
                    vision_features, language_features, current_memory
                )

                # Convert to action command
                cmd = self.convert_vla_output_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Update memory with current experience
                self.update_memory(vision_features, language_features, cmd)

                # Publish memory status
                memory_status_msg = String()
                memory_status_msg.data = json.dumps({
                    'command': self.current_command,
                    'episodic_memory_size': len(self.episodic_memory),
                    'semantic_memory_size': len(self.semantic_memory),
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    },
                    'memory_confidence': self.estimate_memory_confidence()
                })
                self.memory_status_pub.publish(memory_status_msg)

                self.get_logger().info(
                    f'Memory-Augmented VLA - Command: "{self.current_command}", '
                    f'Episodic Memory: {len(self.episodic_memory)}, '
                    f'Semantic Memory: {len(self.semantic_memory)}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in memory-augmented VLA: {str(e)}')

    def extract_memory_augmented_vision_features(self, image_msg, memory_features):
        """Extract vision features with memory augmentation"""
        try:
            # Convert and preprocess image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features = self.vision_encoder(image_tensor, memory_features)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting memory-augmented vision features: {str(e)}')
            return None

    def extract_memory_augmented_language_features(self, command, memory_features):
        """Extract language features with memory augmentation"""
        try:
            # Tokenize command
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            with torch.no_grad():
                features = self.language_encoder(token_tensor, memory_features)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting memory-augmented language features: {str(e)}')
            return None

    def perform_memory_augmented_vla(self, vision_features, language_features, memory_features):
        """Perform VLA with memory augmentation"""
        try:
            with torch.no_grad():
                action_output = self.vla_policy(vision_features, language_features, memory_features)
                return action_output

        except Exception as e:
            self.get_logger().error(f'Error in memory-augmented VLA: {str(e)}')
            return None

    def update_memory(self, vision_features, language_features, action_cmd):
        """Update all memory systems with current experience"""
        try:
            # Create experience entry
            experience = {
                'timestamp': datetime.now(),
                'vision_features': vision_features.detach().cpu().numpy() if vision_features is not None else None,
                'language_features': language_features.detach().cpu().numpy() if language_features is not None else None,
                'action': [action_cmd.linear.x, action_cmd.angular.z] if action_cmd else [0.0, 0.0],
                'command': self.current_command
            }

            # Add to episodic memory
            self.episodic_memory.append(experience)

            # Maintain memory size limit
            if len(self.episodic_memory) > self.max_episodic_memory:
                self.episodic_memory = self.episodic_memory[-self.max_episodic_memory:]

            # Update semantic memory based on experience patterns
            self.update_semantic_memory(experience)

            # Decay older memories (simplified)
            self.decay_episodic_memories()

        except Exception as e:
            self.get_logger().error(f'Error updating memory: {str(e)}')

    def update_semantic_memory(self, experience):
        """Update semantic memory with learned patterns"""
        try:
            # Extract semantic patterns from experience
            command_words = experience['command'].lower().split()

            for word in command_words:
                if word not in self.semantic_memory:
                    self.semantic_memory[word] = {
                        'frequency': 1,
                        'associated_actions': [experience['action']],
                        'last_used': experience['timestamp']
                    }
                else:
                    self.semantic_memory[word]['frequency'] += 1
                    self.semantic_memory[word]['associated_actions'].append(experience['action'])
                    self.semantic_memory[word]['last_used'] = experience['timestamp']

                    # Maintain reasonable history size
                    if len(self.semantic_memory[word]['associated_actions']) > 100:
                        self.semantic_memory[word]['associated_actions'] = (
                            self.semantic_memory[word]['associated_actions'][-50:]
                        )

        except Exception as e:
            self.get_logger().error(f'Error updating semantic memory: {str(e)}')

    def decay_episodic_memories(self):
        """Apply decay to older episodic memories"""
        try:
            current_time = datetime.now()

            # Simplified decay: remove memories older than 1 hour
            cutoff_time = current_time - timedelta(hours=1)
            self.episodic_memory = [
                exp for exp in self.episodic_memory
                if exp['timestamp'] > cutoff_time
            ]

        except Exception as e:
            self.get_logger().error(f'Error decaying episodic memories: {str(e)}')

    def get_memory_features(self):
        """Get aggregated memory features for attention"""
        try:
            if not self.episodic_memory:
                return None

            # Aggregate recent memory features (simplified)
            recent_experiences = self.episodic_memory[-10:]  # Last 10 experiences

            # Average vision and language features from recent experiences
            vision_features_list = []
            language_features_list = []

            for exp in recent_experiences:
                if exp['vision_features'] is not None:
                    vision_features_list.append(exp['vision_features'])
                if exp['language_features'] is not None:
                    language_features_list.append(exp['language_features'])

            # Create averaged memory features
            if vision_features_list:
                avg_vision = np.mean(vision_features_list, axis=0)
                memory_tensor = torch.FloatTensor(avg_vision).unsqueeze(0)  # [1, feature_dim]
                return memory_tensor
            else:
                return None

        except Exception as e:
            self.get_logger().error(f'Error getting memory features: {str(e)}')
            return None

    def estimate_memory_confidence(self):
        """Estimate confidence based on memory availability and relevance"""
        try:
            # Calculate memory confidence based on:
            # 1. Amount of relevant experience
            # 2. Recency of experience
            # 3. Similarity to past experiences

            if not self.episodic_memory:
                return 0.3  # Low confidence without memory

            # Calculate average recency of memories
            current_time = datetime.now()
            total_recency = 0
            for exp in self.episodic_memory:
                time_diff = (current_time - exp['timestamp']).total_seconds()
                # Weight more recent memories higher (inverse relationship)
                total_recency += 1.0 / (1.0 + time_diff / 60.0)  # Normalize by minute

            avg_recency = total_recency / len(self.episodic_memory)

            # Memory confidence based on amount and recency
            memory_amount_factor = min(1.0, len(self.episodic_memory) / 50.0)  # Normalize by 50 experiences
            recency_factor = min(1.0, avg_recency)

            confidence = 0.5 * memory_amount_factor + 0.5 * recency_factor
            return min(1.0, confidence)

        except Exception as e:
            self.get_logger().error(f'Error estimating memory confidence: {str(e)}')
            return 0.5

    def convert_vla_output_to_command(self, action_output):
        """Convert VLA output to robot command"""
        if action_output is None:
            return None

        try:
            action_values = action_output[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting VLA output to command: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    memory_vla = MemoryAugmentedVLANode()

    try:
        rclpy.spin(memory_vla)
    except KeyboardInterrupt:
        memory_vla.get_logger().info('Memory-augmented VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        memory_vla.action_pub.publish(cmd)

        memory_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Embodied VLA Agents

Implementing end-to-end trainable embodied VLA agents:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json

class EmbodiedVLAAgent(Node):
    def __init__(self):
        super().__init__('embodied_vla_agent')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            10
        )

        self.reward_sub = self.create_subscription(
            Float32,
            '/environment_reward',
            self.reward_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.agent_status_pub = self.create_publisher(
            String,
            '/embodied_vla_status',
            10
        )

        # Initialize embodied VLA agent
        self.vla_agent = None
        self.optimizer = None
        self.initialize_embodied_agent()

        # Training state
        self.current_observation = None
        self.current_action = None
        self.current_reward = 0.0
        self.current_command = None
        self.episode_buffer = []
        self.episode_reward = 0.0

        # Training parameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        self.update_frequency = 10  # Update every 10 steps
        self.step_count = 0

        # Control timer
        self.agent_timer = self.create_timer(0.1, self.embodied_vla_loop)

        self.get_logger().info('Embodied VLA agent initialized')

    def initialize_embodied_agent(self):
        """Initialize end-to-end trainable embodied VLA agent"""
        try:
            # Vision encoder
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 32, 7, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 5, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((8, 8))
                    )
                    self.fc = nn.Linear(256 * 8 * 8, 512)

                def forward(self, x):
                    features = self.conv_layers(x)
                    features = features.view(features.size(0), -1)
                    features = F.relu(self.fc(features))
                    return features

            # Language encoder
            class LanguageEncoder(nn.Module):
                def __init__(self):
                    super(LanguageEncoder, self).__init__()
                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True)
                    self.fc = nn.Linear(256, 256)

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    features = self.fc(lstm_out[:, -1, :])  # Use last output
                    return features

            # LiDAR encoder
            class LidarEncoder(nn.Module):
                def __init__(self):
                    super(LidarEncoder, self).__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                def forward(self, x):
                    return self.fc(x)

            # Multimodal fusion
            class MultimodalFusion(nn.Module):
                def __init__(self):
                    super(MultimodalFusion, self).__init__()
                    self.fusion = nn.Sequential(
                        nn.Linear(512 + 256 + 256, 512),  # Vision + Language + LiDAR
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128)
                    )

                def forward(self, vision_features, language_features, lidar_features):
                    combined = torch.cat([vision_features, language_features, lidar_features], dim=1)
                    return self.fusion(combined)

            # Actor-Critic network for RL
            class ActorCritic(nn.Module):
                def __init__(self):
                    super(ActorCritic, self).__init__()

                    # Actor (policy network)
                    self.actor = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 2)  # [linear_x, angular_z]
                    )

                    # Critic (value network)
                    self.critic = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)  # Value
                    )

                def forward(self, fused_features):
                    action_probs = self.actor(fused_features)
                    value = self.critic(fused_features)
                    return action_probs, value

            # Complete VLA agent
            class VLAAgent(nn.Module):
                def __init__(self):
                    super(VLAAgent, self).__init__()
                    self.vision_encoder = VisionEncoder()
                    self.language_encoder = LanguageEncoder()
                    self.lidar_encoder = LidarEncoder()
                    self.multimodal_fusion = MultimodalFusion()
                    self.actor_critic = ActorCritic()

                def forward(self, vision_input, language_input, lidar_input):
                    vision_features = self.vision_encoder(vision_input)
                    language_features = self.language_encoder(language_input)
                    lidar_features = self.lidar_encoder(lidar_input)

                    fused_features = self.multimodal_fusion(
                        vision_features, language_features, lidar_features
                    )

                    action_probs, value = self.actor_critic(fused_features)
                    return action_probs, value

            # Initialize agent
            self.vla_agent = VLAAgent()
            self.optimizer = optim.Adam(self.vla_agent.parameters(), lr=self.learning_rate)

            self.get_logger().info('Embodied VLA agent initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize embodied agent: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def reward_callback(self, msg):
        """Process environment reward"""
        self.current_reward = msg.data

    def embodied_vla_loop(self):
        """Main embodied VLA agent loop"""
        if (self.vla_agent is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Prepare observation
            observation = self.prepare_observation()

            if observation is not None:
                # Get action from agent
                with torch.no_grad():
                    action_probs, value = self.vla_agent(
                        observation['vision'],
                        observation['language'],
                        observation['lidar']
                    )

                # Sample action (with exploration during training)
                action = self.sample_action(action_probs)

                # Convert to robot command
                cmd = self.convert_action_to_command(action)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Store experience for training
                experience = {
                    'observation': observation,
                    'action': action.clone(),
                    'reward': self.current_reward,
                    'value': value.clone()
                }
                self.episode_buffer.append(experience)
                self.episode_reward += self.current_reward

                # Perform learning update periodically
                self.step_count += 1
                if self.step_count % self.update_frequency == 0:
                    self.perform_training_update()

                # Publish agent status
                agent_status_msg = String()
                agent_status_msg.data = json.dumps({
                    'command': self.current_command,
                    'step_count': self.step_count,
                    'episode_reward': self.episode_reward,
                    'episode_buffer_size': len(self.episode_buffer),
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.agent_status_pub.publish(agent_status_msg)

                self.get_logger().info(
                    f'Embodied VLA - Command: "{self.current_command}", '
                    f'Step: {self.step_count}, '
                    f'Episode Reward: {self.episode_reward:.2f}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in embodied VLA agent: {str(e)}')

    def prepare_observation(self):
        """Prepare observation from sensor data"""
        try:
            # Process vision
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            vision_input = torch.FloatTensor(image_tensor)

            # Process language
            tokens = self.current_command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            language_input = torch.LongTensor([token_ids])

            # Process LiDAR
            scan_data = np.array(self.current_scan.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)
            lidar_input = torch.FloatTensor([scan_data])

            return {
                'vision': vision_input,
                'language': language_input,
                'lidar': lidar_input
            }

        except Exception as e:
            self.get_logger().error(f'Error preparing observation: {str(e)}')
            return None

    def sample_action(self, action_probs):
        """Sample action from action probabilities"""
        # Add some exploration noise during training
        noise = torch.randn_like(action_probs) * 0.1  # Exploration factor
        action = action_probs + noise
        return action

    def convert_action_to_command(self, action_tensor):
        """Convert action tensor to robot command"""
        try:
            action_values = action_tensor[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting action to command: {str(e)}')
            return None

    def perform_training_update(self):
        """Perform training update using collected experiences"""
        if len(self.episode_buffer) < 2:
            return

        try:
            # Prepare batch
            batch_vision = []
            batch_language = []
            batch_lidar = []
            batch_actions = []
            batch_rewards = []
            batch_values = []

            for exp in self.episode_buffer:
                batch_vision.append(exp['observation']['vision'])
                batch_language.append(exp['observation']['language'])
                batch_lidar.append(exp['observation']['lidar'])
                batch_actions.append(exp['action'])
                batch_rewards.append(exp['reward'])
                batch_values.append(exp['value'])

            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(batch_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            # Convert to tensors
            batch_vision = torch.cat(batch_vision, dim=0)
            batch_language = torch.cat(batch_language, dim=0)
            batch_lidar = torch.cat(batch_lidar, dim=0)
            batch_actions = torch.cat(batch_actions, dim=0)
            batch_returns = torch.FloatTensor(returns).unsqueeze(1)

            # Forward pass
            action_probs_batch, value_batch = self.vla_agent(
                batch_vision, batch_language, batch_lidar
            )

            # Compute losses
            # Actor loss (policy gradient)
            advantages = batch_returns - value_batch.detach()
            actor_loss = -(action_probs_batch * advantages).mean()

            # Critic loss (value prediction)
            critic_loss = F.mse_loss(value_batch, batch_returns)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vla_agent.parameters(), 40)  # Gradient clipping
            self.optimizer.step()

            # Log training metrics
            self.get_logger().info(
                f'Training Update - Actor Loss: {actor_loss.item():.4f}, '
                f'Critic Loss: {critic_loss.item():.4f}, '
                f'Total Loss: {total_loss.item():.4f}'
            )

            # Clear episode buffer
            self.episode_buffer = []
            self.episode_reward = 0.0

        except Exception as e:
            self.get_logger().error(f'Error in training update: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    embodied_agent = EmbodiedVLAAgent()

    try:
        rclpy.spin(embodied_agent)
    except KeyboardInterrupt:
        embodied_agent.get_logger().info('Embodied VLA agent shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        embodied_agent.action_pub.publish(cmd)

        embodied_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced VLA Applications

Implementing advanced VLA applications for complex robotic tasks:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose, PoseStamped
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json

class AdvancedVLAApplications(Node):
    def __init__(self):
        super().__init__('advanced_vla_applications')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.advanced_status_pub = self.create_publisher(
            String,
            '/advanced_vla_status',
            10
        )

        # Initialize advanced VLA components
        self.advanced_vla_model = None
        self.scene_understanding = None
        self.long_horizon_planner = None
        self.initialize_advanced_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_map = None
        self.current_goal = None

        # Advanced application state
        self.application_state = {
            'current_task': 'idle',
            'task_progress': 0.0,
            'object_tracking': {},
            'navigation_context': {},
            'social_interaction': {}
        }

        # Control timer
        self.advanced_timer = self.create_timer(0.1, self.advanced_vla_loop)

        self.get_logger().info('Advanced VLA applications initialized')

    def initialize_advanced_components(self):
        """Initialize advanced VLA components"""
        try:
            # Advanced scene understanding network
            class AdvancedSceneUnderstanding(nn.Module):
                def __init__(self):
                    super(AdvancedSceneUnderstanding, self).__init__()

                    # Vision processing
                    self.vision_backbone = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((7, 7))
                    )

                    # Object detection head
                    self.object_detection = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 4 * 80)  # 4 coordinates * 80 classes
                    )

                    # Scene classification head
                    self.scene_classifier = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)  # 10 scene types
                    )

                    # Spatial relationship encoder
                    self.spatial_encoder = nn.Sequential(
                        nn.Linear(512 * 7 * 7 + 128, 512),  # Vision + scene context
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128)  # Spatial relationships
                    )

                def forward(self, vision_input):
                    features = self.vision_backbone(vision_input)
                    features_flat = features.view(features.size(0), -1)

                    # Object detection
                    obj_detections = self.object_detection(features_flat)
                    obj_detections = obj_detections.view(obj_detections.size(0), 80, 4)  # [batch, classes, 4 coords]

                    # Scene classification
                    scene_probs = self.scene_classifier(features_flat)

                    # Spatial relationships
                    scene_context = F.softmax(scene_probs, dim=1)
                    spatial_input = torch.cat([features_flat, scene_context], dim=1)
                    spatial_output = self.spatial_encoder(spatial_input)

                    return obj_detections, scene_probs, spatial_output

            # Long-horizon planning network
            class LongHorizonPlanner(nn.Module):
                def __init__(self):
                    super(LongHorizonPlanner, self).__init__()

                    # Language understanding
                    self.lang_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # Map processing
                    self.map_encoder = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((8, 8))
                    )

                    # Planning network
                    self.planning_net = nn.Sequential(
                        nn.Linear(256 + 64 * 8 * 8 + 128, 512),  # Lang + Map + Spatial
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z] per subgoal
                    )

                    # Subgoal generation
                    self.subgoal_generator = nn.Sequential(
                        nn.Linear(256 + 64 * 8 * 8 + 128, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64)  # [x, y, theta] per subgoal
                    )

                def forward(self, language_input, map_input, spatial_context):
                    # Process language
                    lang_embedded = self.lang_encoder[0](language_input)
                    lang_lstm_out, _ = self.lang_encoder[1](lang_embedded)
                    lang_features = self.lang_encoder[2](lang_lstm_out[:, -1, :])

                    # Process map
                    map_features = self.map_encoder(map_input)
                    map_features = map_features.view(map_features.size(0), -1)

                    # Combine inputs
                    combined_input = torch.cat([lang_features, map_features, spatial_context], dim=1)

                    # Generate planning output
                    planning_output = self.planning_net(combined_input)
                    subgoals = self.subgoal_generator(combined_input)

                    return planning_output, subgoals

            # Social interaction network
            class SocialInteractionNetwork(nn.Module):
                def __init__(self):
                    super(SocialInteractionNetwork, self).__init__()

                    # Person detection and tracking
                    self.person_detector = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 4)  # [x, y, w, h] for person bbox
                    )

                    # Social behavior prediction
                    self.social_behavior = nn.Sequential(
                        nn.Linear(512 * 7 * 7 + 4, 256),  # Vision + Person location
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),  # Social interaction type
                        nn.Softmax(dim=1)
                    )

                    # Personal space maintenance
                    self.personal_space = nn.Sequential(
                        nn.Linear(512 * 7 * 7 + 4 + 64, 128),  # Vision + Person + Social
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z] for social navigation
                    )

                def forward(self, vision_input, person_location):
                    features = vision_input.view(vision_input.size(0), -1)

                    # Detect person
                    person_bbox = self.person_detector(features)

                    # Predict social behavior
                    social_input = torch.cat([features, person_location], dim=1)
                    social_probs = self.social_behavior(social_input)

                    # Generate social navigation command
                    social_input_combined = torch.cat([features, person_location, social_probs], dim=1)
                    social_command = self.personal_space(social_input_combined)

                    return person_bbox, social_probs, social_command

            # Initialize models
            self.scene_understanding = AdvancedSceneUnderstanding()
            self.long_horizon_planner = LongHorizonPlanner()
            self.social_network = SocialInteractionNetwork()

            # Set to evaluation mode
            self.scene_understanding.eval()
            self.long_horizon_planner.eval()
            self.social_network.eval()

            self.get_logger().info('Advanced VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced components: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.current_map = msg

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg

    def advanced_vla_loop(self):
        """Main advanced VLA processing loop"""
        if (self.scene_understanding is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Perform scene understanding
            scene_analysis = self.perform_scene_understanding()

            if scene_analysis:
                # Determine application based on command and scene
                application_type = self.determine_application_type(
                    self.current_command, scene_analysis
                )

                # Execute appropriate advanced application
                action_cmd = self.execute_advanced_application(
                    application_type, scene_analysis
                )

                if action_cmd is not None:
                    self.action_pub.publish(action_cmd)

                # Update application state
                self.application_state['current_task'] = application_type

                # Publish advanced status
                advanced_status_msg = String()
                advanced_status_msg.data = json.dumps({
                    'application_type': application_type,
                    'command': self.current_command,
                    'scene_analysis': {
                        'detected_objects': len(scene_analysis['objects']) if scene_analysis['objects'] else 0,
                        'scene_type': scene_analysis['scene_type'] if scene_analysis['scene_type'] else 'unknown'
                    },
                    'action': {
                        'linear_x': float(action_cmd.linear.x) if action_cmd else 0.0,
                        'angular_z': float(action_cmd.angular.z) if action_cmd else 0.0
                    }
                })
                self.advanced_status_pub.publish(advanced_status_msg)

                self.get_logger().info(
                    f'Advanced VLA Application - Type: {application_type}, '
                    f'Command: "{self.current_command}", '
                    f'Scene Objects: {len(scene_analysis["objects"]) if scene_analysis["objects"] else 0}, '
                    f'Action - Linear: {action_cmd.linear.x:.2f}, Angular: {action_cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in advanced VLA applications: {str(e)}')

    def perform_scene_understanding(self):
        """Perform advanced scene understanding"""
        try:
            # Convert image to tensor
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                obj_detections, scene_probs, spatial_output = self.scene_understanding(image_tensor)

                # Process detections
                detected_objects = []
                obj_probs = F.softmax(obj_detections[0], dim=0)  # Apply softmax to get probabilities
                max_probs, max_indices = torch.max(obj_probs, dim=1)

                for i in range(min(10, len(max_probs))):  # Top 10 detections
                    if max_probs[i] > 0.3:  # Confidence threshold
                        detected_objects.append({
                            'class': f'class_{max_indices[i].item()}',
                            'confidence': float(max_probs[i]),
                            'bbox': [0, 0, 0, 0]  # Placeholder - would use actual bbox coords
                        })

                # Determine scene type
                scene_type_idx = torch.argmax(scene_probs[0]).item()
                scene_types = ['office', 'kitchen', 'living_room', 'outdoor', 'corridor',
                              'bedroom', 'garden', 'workshop', 'classroom', 'hallway']
                scene_type = scene_types[scene_type_idx] if scene_type_idx < len(scene_types) else 'unknown'

                return {
                    'objects': detected_objects,
                    'scene_type': scene_type,
                    'spatial_context': spatial_output[0].numpy()
                }

        except Exception as e:
            self.get_logger().error(f'Error in scene understanding: {str(e)}')
            return None

    def determine_application_type(self, command: str, scene_analysis: Dict) -> str:
        """Determine which advanced application to execute"""
        command_lower = command.lower()

        # Check for navigation-related commands
        if any(word in command_lower for word in ['navigate', 'go to', 'move to', 'explore', 'find']):
            return 'long_horizon_navigation'

        # Check for object interaction commands
        elif any(word in command_lower for word in ['approach', 'grasp', 'pickup', 'take', 'get', 'fetch']):
            return 'object_interaction'

        # Check for social interaction commands
        elif any(word in command_lower for word in ['follow', 'accompany', 'escort', 'meet', 'greet']):
            return 'social_interaction'

        # Check for complex task commands
        elif any(word in command_lower for word in ['clean', 'organize', 'assemble', 'disassemble']):
            return 'complex_manipulation'

        # Default based on scene
        else:
            scene_type = scene_analysis.get('scene_type', 'unknown')
            if scene_type in ['office', 'corridor', 'hallway']:
                return 'navigation'
            elif scene_type in ['kitchen', 'living_room', 'bedroom']:
                return 'domestic_assistance'
            else:
                return 'basic_navigation'

    def execute_advanced_application(self, app_type: str, scene_analysis: Dict):
        """Execute specific advanced application"""
        cmd = Twist()

        if app_type == 'long_horizon_navigation':
            cmd = self.execute_long_horizon_navigation(scene_analysis)

        elif app_type == 'object_interaction':
            cmd = self.execute_object_interaction(scene_analysis)

        elif app_type == 'social_interaction':
            cmd = self.execute_social_interaction(scene_analysis)

        elif app_type == 'complex_manipulation':
            cmd = self.execute_complex_manipulation(scene_analysis)

        elif app_type == 'domestic_assistance':
            cmd = self.execute_domestic_assistance(scene_analysis)

        else:
            # Default navigation behavior
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

    def execute_long_horizon_navigation(self, scene_analysis: Dict):
        """Execute long-horizon navigation task"""
        cmd = Twist()

        # In a real implementation, this would use the long_horizon_planner
        # For this example, we'll implement a simplified version

        # Analyze scene for navigation-relevant information
        objects = scene_analysis.get('objects', [])
        scene_type = scene_analysis.get('scene_type', 'unknown')

        # Determine navigation strategy based on scene
        if scene_type == 'corridor':
            # In corridors, move forward but be prepared to turn
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0

            # Check for obstacles
            if self.current_scan:
                ranges = np.array(self.current_scan.ranges)
                valid_ranges = ranges[np.isfinite(ranges)]
                if len(valid_ranges) > 0 and np.min(valid_ranges) < 0.8:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.3  # Turn to avoid

        elif scene_type in ['office', 'living_room']:
            # More complex navigation with obstacle avoidance
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

            # Look for pathways between furniture
            if objects:
                # Simplified obstacle avoidance
                for obj in objects:
                    if obj['confidence'] > 0.5:
                        # Adjust path to avoid object
                        cmd.angular.z += 0.1  # Gentle turn

        else:
            # Default navigation
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        return cmd

    def execute_object_interaction(self, scene_analysis: Dict):
        """Execute object interaction task"""
        cmd = Twist()

        # Look for manipulable objects
        objects = scene_analysis.get('objects', [])
        manipulable_objects = [
            obj for obj in objects
            if obj['confidence'] > 0.7 and
            obj['class'] in ['object', 'box', 'ball', 'cup', 'book']
        ]

        if manipulable_objects:
            # Approach the first manipulable object
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # No manipulable objects found, move forward slowly
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0

        return cmd

    def execute_social_interaction(self, scene_analysis: Dict):
        """Execute social interaction task"""
        cmd = Twist()

        # Look for people in the scene
        people = [
            obj for obj in scene_analysis.get('objects', [])
            if obj['confidence'] > 0.7 and obj['class'] == 'person'
        ]

        if people:
            # Maintain appropriate social distance
            cmd.linear.x = 0.1  # Move slowly toward person
            cmd.angular.z = 0.0
        else:
            # No people detected, continue exploration
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        return cmd

    def execute_complex_manipulation(self, scene_analysis: Dict):
        """Execute complex manipulation task"""
        cmd = Twist()

        # For complex manipulation, move carefully and look for appropriate objects
        cmd.linear.x = 0.1  # Move slowly for precision
        cmd.angular.z = 0.0

        # In a real implementation, this would involve complex manipulation planning
        # based on scene analysis and object affordances

        return cmd

    def execute_domestic_assistance(self, scene_analysis: Dict):
        """Execute domestic assistance task"""
        cmd = Twist()

        # In domestic environments, move with care and look for household objects
        cmd.linear.x = 0.2  # Moderate speed for safety
        cmd.angular.z = 0.0

        # Look for common household objects
        objects = scene_analysis.get('objects', [])
        household_objects = [
            obj for obj in objects
            if obj['confidence'] > 0.6 and
            obj['class'] in ['object', 'box', 'cup', 'book', 'chair', 'table']
        ]

        if household_objects:
            # Adjust behavior based on detected objects
            cmd.linear.x = 0.15  # Slower when objects detected

        return cmd

def main(args=None):
    rclpy.init(args=args)
    advanced_vla = AdvancedVLAApplications()

    try:
        rclpy.spin(advanced_vla)
    except KeyboardInterrupt:
        advanced_vla.get_logger().info('Advanced VLA applications shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_vla.action_pub.publish(cmd)

        advanced_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Advanced VLA Systems

1. **Real-time Performance**: Optimize neural networks for real-time inference
2. **Robustness**: Handle missing or corrupted sensor data gracefully
3. **Safety**: Implement safety checks around all AI-driven decisions
4. **Memory Management**: Efficiently manage GPU and system memory for continuous operation
5. **Modularity**: Design components to be modular and reusable
6. **Validation**: Continuously validate AI outputs against safety constraints
7. **Scalability**: Design systems that can scale to multiple robots or tasks
8. **Interpretability**: Provide explanations for AI decisions when possible

### Physical Grounding and Simulation-to-Real Mapping

When implementing advanced VLA systems:

- **Hardware Acceleration**: Ensure real hardware has compatible NVIDIA GPUs for Isaac ROS optimizations
- **Latency Requirements**: Account for processing delays in real-time systems
- **Sensor Calibration**: Maintain accurate calibration between simulation and real sensors
- **Environmental Differences**: Account for lighting, texture, and material differences
- **Resource Constraints**: Consider computational and memory limitations on real hardware
- **Safety Systems**: Implement proper safety mechanisms around AI decisions
- **Performance Monitoring**: Monitor AI system performance in real environments

### Troubleshooting Advanced VLA Issues

Common advanced VLA problems and solutions:

- **Performance Issues**: Profile neural network inference and optimize
- **Memory Problems**: Monitor GPU memory usage and implement memory management
- **Integration Issues**: Verify data format compatibility between components
- **Training Data Bias**: Ensure training data represents real-world conditions
- **Real-time Violations**: Optimize algorithms and consider hardware upgrades
- **Safety Violations**: Implement additional safety checks and validation layers

### Summary

This chapter explored advanced topics in Vision-Language-Action (VLA) systems for robotics, including large-scale VLA models, hierarchical systems, memory-augmented architectures, embodied agents, and complex applications. You learned about implementing sophisticated VLA systems that can handle long-horizon tasks, maintain context across interactions, and perform complex robotic behaviors. Advanced VLA systems enable robots to understand and execute complex commands that require integration of perception, reasoning, and action in a coherent manner. These systems form the foundation for autonomous robots capable of complex, goal-directed behavior. In the next chapter, we'll explore the integration of all VLA components into complete robotic systems.