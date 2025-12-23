# Module 4: Vision–Language–Action (VLA)

## Chapter 4.6: Multimodal Fusion for Integrated Perception and Action

This chapter explores multimodal fusion techniques that integrate information from multiple sensory modalities (vision, language, audio, touch, etc.) to create comprehensive understanding and coordinated action in robotic systems. Multimodal fusion enables robots to make more informed decisions by combining complementary information from different sensors.

### Understanding Multimodal Fusion

Multimodal fusion combines information from multiple sensory modalities to create a more complete and robust understanding of the environment than any single modality could provide. Key aspects include:

- **Early Fusion**: Combining raw sensory data at the lowest level
- **Late Fusion**: Combining high-level features or decisions from individual modalities
- **Intermediate Fusion**: Combining at intermediate processing levels
- **Decision-Level Fusion**: Combining final decisions from different modalities
- **Cross-Modal Attention**: Attending to relevant information across modalities

### Multimodal Fusion Architecture

The multimodal fusion architecture typically follows this structure:

```
+-------------------+    +-------------------+    +-------------------+
|   Vision Module   |    |   Language Module |    |   Action Module   |
|   (Images, Video) |    |   (Text, Speech)  |    |   (Commands,      |
|                   |    |                   |    |    Motions)      |
+-------------------+    +-------------------+    +-------------------+
           |                       |                       |
           v                       v                       v
+---------------------------------------------------------------+
|                   Feature Extractors                          |
|  (CNN, Transformer, RNN, etc.)                               |
+---------------------------------------------------------------+
           |                       |                       |
           v                       v                       v
+---------------------------------------------------------------+
|                   Fusion Layer                                |
|  (Attention, Concatenation, Weighted Combination)             |
+---------------------------------------------------------------+
           |                       |
           v                       v
+-------------------+    +-------------------+
|   Reasoning &     |    |   Action Planning |
|   Decision Making |    |   (Motor Control) |
+-------------------+    +-------------------+
```

### Basic Multimodal Fusion Implementation

Implementing a foundational multimodal fusion system:

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
from typing import Dict, List, Tuple, Optional
import json

class MultimodalFusionNode(Node):
    def __init__(self):
        super().__init__('multimodal_fusion')

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

        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

        self.fused_cmd_pub = self.create_publisher(
            Twist,
            '/fused_cmd_vel',
            10
        )

        self.fusion_status_pub = self.create_publisher(
            String,
            '/fusion_status',
            10
        )

        # Initialize fusion components
        self.vision_processor = None
        self.language_processor = None
        self.fusion_network = None
        self.initialize_fusion_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_voice = None
        self.current_audio = None
        self.fusion_confidence = 0.0

        # Fusion weights (for weighted combination)
        self.modality_weights = {
            'vision': 0.4,
            'language': 0.3,
            'lidar': 0.3
        }

        # Control timer
        self.fusion_timer = self.create_timer(0.1, self.multimodal_fusion_loop)

        self.get_logger().info('Multimodal fusion node initialized')

    def initialize_fusion_components(self):
        """Initialize multimodal fusion components"""
        try:
            # Vision feature extractor
            class VisionFeatureExtractor(nn.Module):
                def __init__(self):
                    super(VisionFeatureExtractor, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
                    self.fc = nn.Linear(256 * 4 * 4, 256)

                def forward(self, x):
                    features = self.conv_layers(x)
                    features = features.view(features.size(0), -1)
                    features = self.fc(features)
                    return features

            # Language feature extractor (simplified)
            class LanguageFeatureExtractor(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=256):
                    super(LanguageFeatureExtractor, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, 256, batch_first=True)
                    self.fc = nn.Linear(256, 256)

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    # Use last output
                    last_output = lstm_out[:, -1, :]
                    features = self.fc(last_output)
                    return features

            # LiDAR feature extractor
            class LidarFeatureExtractor(nn.Module):
                def __init__(self, input_size=360):
                    super(LidarFeatureExtractor, self).__init__()
                    self.fc1 = nn.Linear(input_size, 256)
                    self.fc2 = nn.Linear(256, 256)
                    self.fc3 = nn.Linear(256, 256)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            # Fusion network
            class FusionNetwork(nn.Module):
                def __init__(self, feature_dim=256, num_modalities=3):
                    super(FusionNetwork, self).__init__()
                    self.num_modalities = num_modalities
                    self.feature_dim = feature_dim

                    # Attention mechanism for modality weighting
                    self.attention = nn.MultiheadAttention(
                        embed_dim=feature_dim,
                        num_heads=8,
                        batch_first=True
                    )

                    # Fusion layers
                    self.fusion_layer = nn.Sequential(
                        nn.Linear(feature_dim * num_modalities, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # Output: linear and angular velocities
                    )

                def forward(self, vision_features, language_features, lidar_features):
                    # Combine features from all modalities
                    combined_features = torch.cat([
                        vision_features.unsqueeze(1),      # [batch, 1, feature_dim]
                        language_features.unsqueeze(1),    # [batch, 1, feature_dim]
                        lidar_features.unsqueeze(1)        # [batch, 1, feature_dim]
                    ], dim=1)  # [batch, num_modalities, feature_dim]

                    # Apply attention to weigh modalities
                    attended_features, attention_weights = self.attention(
                        query=combined_features,
                        key=combined_features,
                        value=combined_features
                    )

                    # Flatten and pass through fusion network
                    flattened = attended_features.view(-1, self.feature_dim * self.num_modalities)
                    output = self.fusion_layer(flattened)

                    return output, attention_weights

            # Initialize models
            self.vision_extractor = VisionFeatureExtractor()
            self.language_extractor = LanguageFeatureExtractor()
            self.lidar_extractor = LidarFeatureExtractor()
            self.fusion_network = FusionNetwork()

            # Set models to evaluation mode
            self.vision_extractor.eval()
            self.language_extractor.eval()
            self.lidar_extractor.eval()
            self.fusion_network.eval()

            self.get_logger().info('Multimodal fusion components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize fusion components: {str(e)}')

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

    def audio_callback(self, msg):
        """Process audio data"""
        self.current_audio = msg

    def multimodal_fusion_loop(self):
        """Main multimodal fusion processing loop"""
        if (self.fusion_network is None or
            self.current_image is None or
            self.current_scan is None or
            self.current_voice is None):
            return

        try:
            # Extract features from each modality
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.extract_language_features(self.current_voice)
            lidar_features = self.extract_lidar_features(self.current_scan)

            # Perform multimodal fusion
            if all(feat is not None for feat in [vision_features, language_features, lidar_features]):
                fused_action, attention_weights = self.fuse_modalities(
                    vision_features, language_features, lidar_features
                )

                # Convert to robot command
                cmd = self.convert_fused_action_to_command(fused_action)

                if cmd is not None:
                    self.fused_cmd_pub.publish(cmd)

                # Calculate fusion confidence
                confidence = self.calculate_fusion_confidence(attention_weights)
                self.fusion_confidence = confidence

                # Publish fusion status
                status_msg = String()
                status_msg.data = json.dumps({
                    'fusion_confidence': confidence,
                    'modality_contributions': {
                        'vision': float(attention_weights[0, 0, 0]) if attention_weights is not None else 0.0,
                        'language': float(attention_weights[0, 0, 1]) if attention_weights is not None else 0.0,
                        'lidar': float(attention_weights[0, 0, 2]) if attention_weights is not None else 0.0
                    },
                    'action': {
                        'linear_x': cmd.linear.x if cmd else 0.0,
                        'angular_z': cmd.angular.z if cmd else 0.0
                    }
                })
                self.fusion_status_pub.publish(status_msg)

                self.get_logger().info(
                    f'Multimodal Fusion - Vision: {vision_features.shape if vision_features is not None else "None"}, '
                    f'Language: {language_features.shape if language_features is not None else "None"}, '
                    f'LiDAR: {lidar_features.shape if lidar_features is not None else "None"}, '
                    f'Confidence: {confidence:.3f}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in multimodal fusion: {str(e)}')

    def extract_vision_features(self, image_msg):
        """Extract features from camera image"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Preprocess image
            image_resized = cv2.resize(cv_image, (64, 64))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
            image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features = self.vision_extractor(image_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting vision features: {str(e)}')
            return None

    def extract_language_features(self, voice_command):
        """Extract features from voice command"""
        try:
            # Simple tokenization (in real implementation, use proper tokenizer)
            tokens = voice_command.lower().split()
            # Convert to token IDs using simple hashing
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad or truncate to fixed length
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])  # [1, max_length]

            with torch.no_grad():
                features = self.language_extractor(token_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting language features: {str(e)}')
            return None

    def extract_lidar_features(self, scan_msg):
        """Extract features from laser scan"""
        try:
            # Process scan data
            ranges = np.array(scan_msg.ranges)
            ranges = np.nan_to_num(ranges, nan=3.0)  # Replace NaN with max range
            ranges = np.clip(ranges, 0.0, 3.0)  # Clip to max range

            # Normalize
            ranges_normalized = ranges / 3.0

            # Convert to tensor
            scan_tensor = torch.FloatTensor(ranges_normalized).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                features = self.lidar_extractor(scan_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting LiDAR features: {str(e)}')
            return None

    def fuse_modalities(self, vision_features, language_features, lidar_features):
        """Fuse features from different modalities"""
        try:
            with torch.no_grad():
                fused_output, attention_weights = self.fusion_network(
                    vision_features, language_features, lidar_features
                )

                return fused_output, attention_weights

        except Exception as e:
            self.get_logger().error(f'Error in multimodal fusion: {str(e)}')
            return None, None

    def convert_fused_action_to_command(self, fused_action):
        """Convert fused action to robot command"""
        if fused_action is None:
            return None

        try:
            action_values = fused_action[0].numpy()  # Remove batch dimension

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting fused action: {str(e)}')
            return None

    def calculate_fusion_confidence(self, attention_weights):
        """Calculate confidence in fusion result"""
        if attention_weights is None:
            return 0.5  # Default confidence

        try:
            # Calculate entropy of attention weights as uncertainty measure
            weights_flat = attention_weights[0, 0, :].numpy()  # Get first sample, first position
            weights_normalized = weights_flat / np.sum(weights_flat)  # Normalize

            # Calculate entropy (higher entropy = more uncertainty = lower confidence)
            entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8))
            max_entropy = np.log(len(weights_normalized))

            # Convert entropy to confidence (lower entropy = higher confidence)
            confidence = 1.0 - (entropy / max_entropy)
            return confidence

        except Exception as e:
            self.get_logger().error(f'Error calculating fusion confidence: {str(e)}')
            return 0.5

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultimodalFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Multimodal fusion node shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        fusion_node.fused_cmd_pub.publish(cmd)

        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Multimodal Fusion with Cross-Modal Attention

Implementing more sophisticated fusion using cross-modal attention mechanisms:

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
from typing import Dict, List, Tuple
import json

class AdvancedMultimodalFusion(Node):
    def __init__(self):
        super().__init__('advanced_multimodal_fusion')

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
            '/fused_cmd_vel',
            10
        )

        self.attention_pub = self.create_publisher(
            String,
            '/cross_modal_attention',
            10
        )

        self.fusion_pub = self.create_publisher(
            String,
            '/advanced_fusion_output',
            10
        )

        # Initialize advanced fusion components
        self.initialize_advanced_fusion_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_voice = None
        self.fusion_results = None

        # Control timer
        self.advanced_fusion_timer = self.create_timer(0.1, self.advanced_fusion_loop)

        self.get_logger().info('Advanced multimodal fusion initialized')

    def initialize_advanced_fusion_components(self):
        """Initialize advanced fusion components with cross-modal attention"""
        try:
            # Vision encoder with attention
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()

                    # Convolutional feature extraction
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    )

                    # Spatial attention mechanism
                    self.spatial_attention = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 1, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    features = self.conv_layers(x)

                    # Apply spatial attention
                    attention_weights = self.spatial_attention(features)
                    attended_features = features * attention_weights

                    return attended_features, attention_weights

            # Language encoder with transformer
            class LanguageEncoder(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=256, num_heads=8):
                    super(LanguageEncoder, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)

                    # Self-attention layer for language
                    self.self_attention = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        batch_first=True
                    )

                    self.feed_forward = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.ReLU(),
                        nn.Linear(embed_dim * 4, embed_dim)
                    )
                    self.layer_norm1 = nn.LayerNorm(embed_dim)
                    self.layer_norm2 = nn.LayerNorm(embed_dim)

                def forward(self, x):
                    # x shape: [batch, seq_len]
                    embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

                    # Self-attention
                    attended, _ = self.self_attention(embedded, embedded, embedded)
                    attended = self.layer_norm1(attended + embedded)

                    # Feed forward
                    ff_out = self.feed_forward(attended)
                    output = self.layer_norm2(ff_out + attended)

                    # Average across sequence dimension
                    output = torch.mean(output, dim=1)  # [batch, embed_dim]

                    return output

            # LiDAR encoder with attention
            class LidarEncoder(nn.Module):
                def __init__(self, input_size=360):
                    super(LidarEncoder, self).__init__()

                    # Process LiDAR data with attention
                    self.feature_extractor = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU()
                    )

                    # Attention over LiDAR sectors
                    self.attention = nn.Sequential(
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Softmax(dim=1)
                    )

                def forward(self, x):
                    features = self.feature_extractor(x)  # [batch, 256]

                    # Reshape for sector attention
                    sector_features = features.unsqueeze(1)  # [batch, 1, 256]

                    # Calculate attention over sectors
                    attention_weights = self.attention(features)  # [batch, 1]

                    # Apply attention
                    attended_features = features * attention_weights

                    return attended_features, attention_weights

            # Cross-modal attention module
            class CrossModalAttention(nn.Module):
                def __init__(self, feature_dim=256):
                    super(CrossModalAttention, self).__init__()

                    # Query, key, value projections for each modality
                    self.vision_q = nn.Linear(feature_dim, feature_dim)
                    self.vision_k = nn.Linear(feature_dim, feature_dim)
                    self.vision_v = nn.Linear(feature_dim, feature_dim)

                    self.language_q = nn.Linear(feature_dim, feature_dim)
                    self.language_k = nn.Linear(feature_dim, feature_dim)
                    self.language_v = nn.Linear(feature_dim, feature_dim)

                    self.lidar_q = nn.Linear(feature_dim, feature_dim)
                    self.lidar_k = nn.Linear(feature_dim, feature_dim)
                    self.lidar_v = nn.Linear(feature_dim, feature_dim)

                    # Multi-head attention
                    self.multihead_attn = nn.MultiheadAttention(
                        embed_dim=feature_dim,
                        num_heads=8,
                        batch_first=True
                    )

                    # Output projection
                    self.output_projection = nn.Linear(feature_dim, feature_dim)

                def forward(self, vision_feat, language_feat, lidar_feat):
                    # Project features to query, key, value spaces
                    vision_q = self.vision_q(vision_feat)
                    vision_k = self.vision_k(vision_feat)
                    vision_v = self.vision_v(vision_feat)

                    language_q = self.language_q(language_feat)
                    language_k = self.language_k(language_feat)
                    language_v = self.language_v(language_feat)

                    lidar_q = self.lidar_q(lidar_feat)
                    lidar_k = self.lidar_k(lidar_feat)
                    lidar_v = self.lidar_v(lidar_feat)

                    # Combine all modalities
                    queries = torch.stack([vision_q, language_q, lidar_q], dim=1)  # [batch, 3, feature_dim]
                    keys = torch.stack([vision_k, language_k, lidar_k], dim=1)
                    values = torch.stack([vision_v, language_v, lidar_v], dim=1)

                    # Cross-modal attention
                    attended, attention_weights = self.multihead_attn(queries, keys, values)

                    # Apply residual connection and output projection
                    output = self.output_projection(attended + queries)

                    return output, attention_weights

            # Action decoder
            class ActionDecoder(nn.Module):
                def __init__(self, input_dim=256):
                    super(ActionDecoder, self).__init__()

                    self.decoder = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z]
                    )

                def forward(self, x):
                    # Average across modalities if needed
                    if len(x.shape) > 2:
                        x = torch.mean(x, dim=1)  # Average across modalities
                    output = self.decoder(x)
                    return output

            # Initialize models
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
            self.lidar_encoder = LidarEncoder()
            self.cross_attention = CrossModalAttention()
            self.action_decoder = ActionDecoder()

            # Set to evaluation mode
            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.lidar_encoder.eval()
            self.cross_attention.eval()
            self.action_decoder.eval()

            self.get_logger().info('Advanced fusion components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced fusion: {str(e)}')

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

    def advanced_fusion_loop(self):
        """Main advanced fusion processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_scan is None or
            self.current_voice is None):
            return

        try:
            # Extract features from all modalities
            vision_features, vision_attention = self.extract_vision_features_advanced(self.current_image)
            language_features = self.extract_language_features_advanced(self.current_voice)
            lidar_features, lidar_attention = self.extract_lidar_features_advanced(self.current_scan)

            if all(feat is not None for feat in [vision_features, language_features, lidar_features]):
                # Apply cross-modal attention
                fused_features, cross_attention_weights = self.apply_cross_modal_attention(
                    vision_features, language_features, lidar_features
                )

                # Decode action
                action_output = self.decode_action(fused_features)

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Publish attention weights
                attention_msg = String()
                attention_msg.data = json.dumps({
                    'cross_modal_attention': cross_attention_weights[0].tolist() if cross_attention_weights is not None else [],
                    'vision_attention': vision_attention[0].tolist() if vision_attention is not None else [],
                    'lidar_attention': lidar_attention[0].tolist() if lidar_attention is not None else [],
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.attention_pub.publish(attention_msg)

                # Publish fusion results
                fusion_msg = String()
                fusion_msg.data = json.dumps({
                    'modality_confidence': {
                        'vision': float(torch.mean(vision_attention)) if vision_attention is not None else 0.0,
                        'language': 0.8,  # Default high confidence for language
                        'lidar': float(torch.mean(lidar_attention)) if lidar_attention is not None else 0.0
                    },
                    'fusion_confidence': float(torch.mean(cross_attention_weights)) if cross_attention_weights is not None else 0.5,
                    'command': f'Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                })
                self.fusion_pub.publish(fusion_msg)

                self.get_logger().info(
                    f'Advanced Fusion - Vision: {vision_features.shape if vision_features is not None else "None"}, '
                    f'Language: {language_features.shape if language_features is not None else "None"}, '
                    f'LiDAR: {lidar_features.shape if lidar_features is not None else "None"}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in advanced fusion: {str(e)}')

    def extract_vision_features_advanced(self, image_msg):
        """Extract advanced vision features with spatial attention"""
        try:
            # Convert and preprocess image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
            image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features, attention = self.vision_encoder(image_tensor)

                # Flatten spatial dimensions for fusion
                batch_size, channels, height, width = features.shape
                features = features.view(batch_size, channels, -1)  # [batch, channels, height*width]
                features = torch.mean(features, dim=2)  # Average across spatial dimensions

                return features, attention

        except Exception as e:
            self.get_logger().error(f'Error extracting advanced vision features: {str(e)}')
            return None, None

    def extract_language_features_advanced(self, voice_command):
        """Extract advanced language features with transformer attention"""
        try:
            # Tokenize command
            tokens = voice_command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate to fixed length
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])  # [1, max_length]

            with torch.no_grad():
                features = self.language_encoder(token_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting advanced language features: {str(e)}')
            return None

    def extract_lidar_features_advanced(self, scan_msg):
        """Extract advanced LiDAR features with sector attention"""
        try:
            # Process scan data
            ranges = np.array(scan_msg.ranges)
            ranges = np.nan_to_num(ranges, nan=3.0)
            ranges = np.clip(ranges, 0.0, 3.0)

            # Normalize
            ranges_normalized = ranges / 3.0
            scan_tensor = torch.FloatTensor(ranges_normalized).unsqueeze(0)  # [1, num_ranges]

            with torch.no_grad():
                features, attention = self.lidar_encoder(scan_tensor)
                return features, attention

        except Exception as e:
            self.get_logger().error(f'Error extracting advanced LiDAR features: {str(e)}')
            return None, None

    def apply_cross_modal_attention(self, vision_features, language_features, lidar_features):
        """Apply cross-modal attention to fuse features"""
        try:
            with torch.no_grad():
                fused_features, attention_weights = self.cross_attention(
                    vision_features, language_features, lidar_features
                )
                return fused_features, attention_weights

        except Exception as e:
            self.get_logger().error(f'Error in cross-modal attention: {str(e)}')
            return None, None

    def decode_action(self, fused_features):
        """Decode action from fused features"""
        try:
            with torch.no_grad():
                action_output = self.action_decoder(fused_features)
                return action_output

        except Exception as e:
            self.get_logger().error(f'Error decoding action: {str(e)}')
            return None

    def convert_action_to_command(self, action_output):
        """Convert action output to robot command"""
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

def main(args=None):
    rclpy.init(args=args)
    advanced_fusion = AdvancedMultimodalFusion()

    try:
        rclpy.spin(advanced_fusion)
    except KeyboardInterrupt:
        advanced_fusion.get_logger().info('Advanced multimodal fusion shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_fusion.action_pub.publish(cmd)

        advanced_fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Late Fusion with Confidence Weighting

Implementing late fusion with confidence-based weighting:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json

class LateFusionConfidenceNode(Node):
    def __init__(self):
        super().__init__('late_fusion_confidence')

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

        self.confidence_sub = self.create_subscription(
            Float32,
            '/modality_confidence',
            self.confidence_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/fused_cmd_vel',
            10
        )

        self.confidence_pub = self.create_publisher(
            String,
            '/modality_confidence_output',
            10
        )

        # Initialize late fusion components
        self.vision_processor = None
        self.language_processor = None
        self.lidar_processor = None
        self.confidence_estimator = None
        self.initialize_late_fusion_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_voice = None
        self.modality_confidences = {
            'vision': 0.7,
            'language': 0.9,
            'lidar': 0.8
        }

        # Individual modality outputs
        self.vision_output = None
        self.language_output = None
        self.lidar_output = None

        # Control timer
        self.late_fusion_timer = self.create_timer(0.1, self.late_fusion_loop)

        self.get_logger().info('Late fusion with confidence node initialized')

    def initialize_late_fusion_components(self):
        """Initialize late fusion components"""
        try:
            # Individual modality processors
            class VisionProcessor(nn.Module):
                def __init__(self):
                    super(VisionProcessor, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(128 * 4 * 4, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z]
                    )

                def forward(self, x):
                    features = self.conv_layers(x)
                    features = features.view(features.size(0), -1)
                    output = self.fc(features)
                    return output

            class LanguageProcessor(nn.Module):
                def __init__(self):
                    super(LanguageProcessor, self).__init__()
                    self.embedding = nn.Embedding(10000, 128)
                    self.lstm = nn.LSTM(128, 128, batch_first=True)
                    self.fc = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z]
                    )

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    # Use last output
                    last_output = lstm_out[:, -1, :]
                    output = self.fc(last_output)
                    return output

            class LidarProcessor(nn.Module):
                def __init__(self):
                    super(LidarProcessor, self).__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(360, 256),  # Assuming 360-point scan
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # [linear_x, angular_z]
                    )

                def forward(self, x):
                    output = self.fc(x)
                    return output

            # Confidence estimator
            class ConfidenceEstimator(nn.Module):
                def __init__(self):
                    super(ConfidenceEstimator, self).__init__()
                    self.vision_confidence = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

                    self.language_confidence = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

                    self.lidar_confidence = nn.Sequential(
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

                def forward(self, vision_features, language_features, lidar_features):
                    vision_conf = self.vision_confidence(vision_features)
                    language_conf = self.language_confidence(language_features)
                    lidar_conf = self.lidar_confidence(lidar_features)

                    return vision_conf, language_conf, lidar_conf

            # Initialize models
            self.vision_processor = VisionProcessor()
            self.language_processor = LanguageProcessor()
            self.lidar_processor = LidarProcessor()
            self.confidence_estimator = ConfidenceEstimator()

            # Set to evaluation mode
            self.vision_processor.eval()
            self.language_processor.eval()
            self.lidar_processor.eval()
            self.confidence_estimator.eval()

            self.get_logger().info('Late fusion components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize late fusion: {str(e)}')

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

    def confidence_callback(self, msg):
        """Process external confidence estimates"""
        # In a real system, this could provide additional confidence information
        pass

    def late_fusion_loop(self):
        """Main late fusion processing loop"""
        if (self.vision_processor is None or
            self.current_image is None or
            self.current_scan is None or
            self.current_voice is None):
            return

        try:
            # Process each modality separately
            vision_output = self.process_vision_modality()
            language_output = self.process_language_modality()
            lidar_output = self.process_lidar_modality()

            # Estimate confidence for each modality
            vision_conf, language_conf, lidar_conf = self.estimate_modality_confidence()

            # Update stored confidences
            self.modality_confidences['vision'] = float(vision_conf)
            self.modality_confidences['language'] = float(language_conf)
            self.modality_confidences['lidar'] = float(lidar_conf)

            # Perform weighted fusion based on confidences
            fused_output = self.weighted_fusion(
                vision_output, language_output, lidar_output,
                vision_conf, language_conf, lidar_conf
            )

            # Convert to robot command
            cmd = self.convert_output_to_command(fused_output)

            if cmd is not None:
                self.action_pub.publish(cmd)

            # Publish confidence information
            confidence_msg = String()
            confidence_msg.data = json.dumps({
                'modality_confidences': self.modality_confidences,
                'individual_outputs': {
                    'vision': [float(vision_output[0, 0]), float(vision_output[0, 1])] if vision_output is not None else [0.0, 0.0],
                    'language': [float(language_output[0, 0]), float(language_output[0, 1])] if language_output is not None else [0.0, 0.0],
                    'lidar': [float(lidar_output[0, 0]), float(lidar_output[0, 1])] if lidar_output is not None else [0.0, 0.0]
                },
                'fused_output': [float(cmd.linear.x), float(cmd.angular.z)] if cmd is not None else [0.0, 0.0]
            })
            self.confidence_pub.publish(confidence_msg)

            self.get_logger().info(
                f'Late Fusion - Vision Conf: {vision_conf:.3f}, '
                f'Language Conf: {language_conf:.3f}, '
                f'LiDAR Conf: {lidar_conf:.3f}, '
                f'Fused Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in late fusion: {str(e)}')

    def process_vision_modality(self):
        """Process vision modality independently"""
        try:
            # Convert and preprocess image
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            image_resized = cv2.resize(cv_image, (64, 64))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                output = self.vision_processor(image_tensor)
                return output

        except Exception as e:
            self.get_logger().error(f'Error processing vision modality: {str(e)}')
            return torch.zeros(1, 2)  # Default output

    def process_language_modality(self):
        """Process language modality independently"""
        try:
            # Tokenize command
            tokens = self.current_voice.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids]).unsqueeze(0)  # [1, 1, max_length]

            with torch.no_grad():
                output = self.language_processor(token_tensor)
                return output

        except Exception as e:
            self.get_logger().error(f'Error processing language modality: {str(e)}')
            return torch.zeros(1, 2)  # Default output

    def process_lidar_modality(self):
        """Process LiDAR modality independently"""
        try:
            # Process scan data
            ranges = np.array(self.current_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=3.0)
            ranges = np.clip(ranges, 0.0, 3.0)
            ranges_normalized = ranges / 3.0

            scan_tensor = torch.FloatTensor(ranges_normalized).unsqueeze(0)

            with torch.no_grad():
                output = self.lidar_processor(scan_tensor)
                return output

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR modality: {str(e)}')
            return torch.zeros(1, 2)  # Default output

    def estimate_modality_confidence(self):
        """Estimate confidence for each modality"""
        try:
            # Get features from each processor (simplified - in real implementation, you'd extract intermediate features)
            # For this example, we'll use the current outputs to estimate confidence
            # In a real system, you'd have separate confidence networks that output confidence scores

            # Simulate confidence based on various factors
            vision_conf = self.estimate_vision_confidence()
            language_conf = self.estimate_language_confidence()
            lidar_conf = self.estimate_lidar_confidence()

            return vision_conf, language_conf, lidar_conf

        except Exception as e:
            self.get_logger().error(f'Error estimating modality confidence: {str(e)}')
            return 0.7, 0.9, 0.8  # Default confidences

    def estimate_vision_confidence(self):
        """Estimate vision modality confidence"""
        # In a real implementation, this would analyze image quality, detection certainty, etc.
        # For this example, we'll use a simple heuristic
        if self.current_detections and len(self.current_detections.detections) > 0:
            # Higher confidence if objects are detected
            avg_confidence = np.mean([
                det.results[0].hypothesis.score if det.results else 0.0
                for det in self.current_detections.detections
            ])
            return max(0.3, min(1.0, avg_confidence))
        else:
            # Lower confidence if no objects detected
            return 0.5

    def estimate_language_confidence(self):
        """Estimate language modality confidence"""
        # In a real implementation, this would use ASR confidence, grammar checking, etc.
        # For this example, we'll use a simple heuristic
        if self.current_voice and len(self.current_voice.split()) > 1:
            return 0.9  # High confidence for multi-word commands
        else:
            return 0.6  # Lower confidence for single words

    def estimate_lidar_confidence(self):
        """Estimate LiDAR modality confidence"""
        # In a real implementation, this would analyze scan quality, obstacle density, etc.
        if self.current_scan:
            valid_ranges = [r for r in self.current_scan.ranges if np.isfinite(r)]
            if len(valid_ranges) > len(self.current_scan.ranges) * 0.8:  # 80% valid ranges
                return 0.8  # High confidence for mostly valid scan
            else:
                return 0.5  # Lower confidence for many invalid ranges
        else:
            return 0.3

    def weighted_fusion(self, vision_output, language_output, lidar_output,
                       vision_conf, language_conf, lidar_conf):
        """Perform weighted fusion based on modality confidences"""
        try:
            # Normalize confidences to sum to 1
            total_conf = vision_conf + language_conf + lidar_conf
            if total_conf == 0:
                # If all confidences are zero, use equal weights
                vision_weight = language_weight = lidar_weight = 1.0 / 3.0
            else:
                vision_weight = vision_conf / total_conf
                language_weight = language_conf / total_conf
                lidar_weight = lidar_conf / total_conf

            # Apply weights to outputs
            weighted_vision = vision_output * vision_weight
            weighted_language = language_output * language_weight
            weighted_lidar = lidar_output * lidar_weight

            # Sum weighted outputs
            fused_output = weighted_vision + weighted_language + weighted_lidar

            return fused_output

        except Exception as e:
            self.get_logger().error(f'Error in weighted fusion: {str(e)}')
            # Return average of outputs as fallback
            return (vision_output + language_output + lidar_output) / 3.0

    def convert_output_to_command(self, fused_output):
        """Convert fused output to robot command"""
        if fused_output is None:
            return None

        try:
            output_values = fused_output[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(output_values[0])
            cmd.angular.z = float(output_values[1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting output to command: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    late_fusion = LateFusionConfidenceNode()

    try:
        rclpy.spin(late_fusion)
    except KeyboardInterrupt:
        late_fusion.get_logger().info('Late fusion with confidence shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        late_fusion.action_pub.publish(cmd)

        late_fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Multimodal Scene Understanding

Implementing multimodal scene understanding that combines visual and linguistic information:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json

class MultimodalSceneUnderstanding(Node):
    def __init__(self):
        super().__init__('multimodal_scene_understanding')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
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

        self.scene_pub = self.create_publisher(
            String,
            '/scene_description',
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize scene understanding components
        self.scene_analyzer = None
        self.vision_language_fusion = None
        self.initialize_scene_understanding_components()

        # State variables
        self.current_image = None
        self.current_detections = None
        self.current_command = None
        self.scene_description = None

        # Control timer
        self.scene_timer = self.create_timer(0.1, self.scene_understanding_loop)

        self.get_logger().info('Multimodal scene understanding initialized')

    def initialize_scene_understanding_components(self):
        """Initialize scene understanding components"""
        try:
            # Vision-language fusion model for scene understanding
            class VisionLanguageFusion(nn.Module):
                def __init__(self):
                    super(VisionLanguageFusion, self).__init__()

                    # Vision encoder
                    self.vision_encoder = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((8, 8))
                    )

                    # Language encoder
                    self.language_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # Cross-modal attention
                    self.cross_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    # Scene understanding head
                    self.scene_classifier = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)  # 10 scene categories for example
                    )

                    # Object-attribute association
                    self.attribute_predictor = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 20)  # 20 attribute categories
                    )

                def forward(self, vision_input, language_input, object_features):
                    # Process vision input
                    vision_features = self.vision_encoder(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), vision_features.size(1), -1)
                    vision_features = torch.mean(vision_features, dim=2)  # Average spatial features

                    # Process language input
                    lang_embedded = self.language_encoder[0](language_input)
                    lang_lstm_out, _ = self.language_encoder[1](lang_embedded)
                    lang_features = self.language_encoder[2](lang_lstm_out[:, -1, :])  # Use last output

                    # Cross-modal attention
                    attended_features, attention_weights = self.cross_attention(
                        query=vision_features.unsqueeze(1),
                        key=lang_features.unsqueeze(1),
                        value=object_features
                    )

                    # Scene classification
                    scene_output = self.scene_classifier(attended_features.squeeze(1))

                    # Attribute prediction
                    attribute_output = self.attribute_predictor(attended_features.squeeze(1))

                    return scene_output, attribute_output, attention_weights

            # Scene analyzer
            class SceneAnalyzer:
                def __init__(self, node):
                    self.node = node
                    self.fusion_model = None
                    self.initialize_model()

                def initialize_model(self):
                    """Initialize the vision-language fusion model"""
                    try:
                        self.fusion_model = VisionLanguageFusion()
                        self.fusion_model.eval()
                        self.node.get_logger().info('Scene fusion model initialized')
                    except Exception as e:
                        self.node.get_logger().error(f'Failed to initialize scene model: {str(e)}')

                def analyze_scene(self, image, detections, command):
                    """Analyze scene using vision-language fusion"""
                    if self.fusion_model is None:
                        return self.fallback_scene_analysis(detections, command)

                    try:
                        # Prepare vision input
                        cv_image = self.node.bridge.imgmsg_to_cv2(image, "bgr8")
                        image_resized = cv2.resize(cv_image, (64, 64))
                        image_normalized = image_resized.astype(np.float32) / 255.0
                        image_tensor = np.transpose(image_normalized, (2, 0, 1))
                        image_tensor = np.expand_dims(image_tensor, axis=0)
                        image_tensor = torch.FloatTensor(image_tensor)

                        # Prepare language input
                        tokens = command.lower().split()
                        token_ids = [hash(token) % 10000 for token in tokens]

                        max_length = 20
                        if len(token_ids) < max_length:
                            token_ids.extend([0] * (max_length - len(token_ids)))
                        else:
                            token_ids = token_ids[:max_length]

                        language_tensor = torch.LongTensor([token_ids])

                        # Prepare object features from detections
                        object_features = self.extract_object_features(detections)

                        with torch.no_grad():
                            scene_output, attribute_output, attention_weights = self.fusion_model(
                                image_tensor, language_tensor, object_features
                            )

                        # Process outputs
                        scene_categories = ['office', 'kitchen', 'living_room', 'bedroom',
                                          'corridor', 'outdoor', 'garden', 'workshop',
                                          'classroom', 'hallway']
                        scene_probs = F.softmax(scene_output, dim=1)
                        predicted_scene_idx = torch.argmax(scene_probs, dim=1).item()
                        predicted_scene = scene_categories[predicted_scene_idx]

                        # Extract object-attribute associations
                        attribute_categories = ['red', 'blue', 'large', 'small', 'moving',
                                              'stationary', 'obstacle', 'person', 'object', 'furniture']
                        attribute_probs = F.softmax(attribute_output, dim=1)

                        # Create scene description
                        scene_description = {
                            'predicted_scene': predicted_scene,
                            'scene_confidence': float(torch.max(scene_probs).item()),
                            'objects': self.format_detections(detections),
                            'command': command,
                            'object_attributes': self.associate_objects_attributes(detections, attribute_probs),
                            'spatial_relationships': self.infer_spatial_relationships(detections)
                        }

                        return scene_description

                    except Exception as e:
                        self.node.get_logger().error(f'Error in scene analysis: {str(e)}')
                        return self.fallback_scene_analysis(detections, command)

                def extract_object_features(self, detections):
                    """Extract features from object detections"""
                    if not detections or not detections.detections:
                        # Return default features
                        return torch.zeros(1, 1, 256)

                    features = []
                    for detection in detections.detections:
                        # Simple feature representation based on bounding box and class
                        bbox = detection.bbox
                        class_id = detection.results[0].hypothesis.class_id if detection.results else 'unknown'

                        # Create simple feature vector
                        feature_vector = torch.tensor([
                            bbox.center.x, bbox.center.y,  # Position
                            bbox.size_x, bbox.size_y,     # Size
                            hash(class_id) % 1000 / 1000.0,  # Class (hashed to [0,1])
                            detection.results[0].hypothesis.score if detection.results else 0.0  # Confidence
                        ], dtype=torch.float32)

                        features.append(feature_vector)

                    if features:
                        # Stack features and add batch dimension
                        stacked_features = torch.stack(features)
                        return stacked_features.unsqueeze(0)  # [batch, num_objects, features]
                    else:
                        return torch.zeros(1, 1, 6)  # Default: [batch, 1, 6]

                def format_detections(self, detections):
                    """Format detections for scene description"""
                    if not detections:
                        return []

                    formatted_objects = []
                    for detection in detections.detections:
                        obj_info = {
                            'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                            'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                            'position': {
                                'x': detection.bbox.center.x,
                                'y': detection.bbox.center.y
                            },
                            'size': {
                                'width': detection.bbox.size_x,
                                'height': detection.bbox.size_y
                            }
                        }
                        formatted_objects.append(obj_info)

                    return formatted_objects

                def associate_objects_attributes(self, detections, attribute_probs):
                    """Associate detected objects with predicted attributes"""
                    if not detections or not detections.detections:
                        return []

                    attribute_categories = ['red', 'blue', 'large', 'small', 'moving',
                                          'stationary', 'obstacle', 'person', 'object', 'furniture']

                    object_attributes = []
                    for i, detection in enumerate(detections.detections):
                        if i < attribute_probs.size(0):
                            attr_probs = attribute_probs[i]
                            top_attrs = torch.topk(attr_probs, 3)  # Top 3 attributes

                            attrs = []
                            for j in range(3):
                                attr_idx = top_attrs.indices[j].item()
                                attr_prob = top_attrs.values[j].item()

                                if attr_prob > 0.3:  # Threshold for inclusion
                                    attrs.append({
                                        'attribute': attribute_categories[attr_idx],
                                        'confidence': attr_prob
                                    })

                            object_attributes.append({
                                'object_class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                                'attributes': attrs
                            })

                    return object_attributes

                def infer_spatial_relationships(self, detections):
                    """Infer spatial relationships between objects"""
                    if not detections or len(detections.detections) < 2:
                        return []

                    relationships = []
                    for i, obj1 in enumerate(detections.detections):
                        for j, obj2 in enumerate(detections.detections):
                            if i != j:
                                # Calculate spatial relationship
                                dx = obj2.bbox.center.x - obj1.bbox.center.x
                                dy = obj2.bbox.center.y - obj1.bbox.center.y
                                distance = np.sqrt(dx*dx + dy*dy)

                                # Determine relationship based on position
                                if abs(dx) > abs(dy):
                                    if dx > 0:
                                        relation = 'right_of'
                                    else:
                                        relation = 'left_of'
                                else:
                                    if dy > 0:
                                        relation = 'below'
                                    else:
                                        relation = 'above'

                                relationships.append({
                                    'object1': obj1.results[0].hypothesis.class_id if obj1.results else 'unknown',
                                    'object2': obj2.results[0].hypothesis.class_id if obj2.results else 'unknown',
                                    'relationship': relation,
                                    'distance': distance
                                })

                    return relationships

                def fallback_scene_analysis(self, detections, command):
                    """Fallback scene analysis when neural model fails"""
                    return {
                        'predicted_scene': 'unknown',
                        'scene_confidence': 0.0,
                        'objects': self.format_detections(detections),
                        'command': command,
                        'object_attributes': [],
                        'spatial_relationships': self.infer_spatial_relationships(detections)
                    }

            self.scene_analyzer = SceneAnalyzer(self)
            self.get_logger().info('Scene analyzer initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize scene understanding: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def scene_understanding_loop(self):
        """Main scene understanding processing loop"""
        if (self.scene_analyzer is None or
            self.current_image is None or
            self.current_detections is None or
            self.current_command is None):
            return

        try:
            # Analyze scene using vision-language fusion
            scene_description = self.scene_analyzer.analyze_scene(
                self.current_image,
                self.current_detections,
                self.current_command
            )

            # Generate action based on scene understanding and command
            action_cmd = self.generate_action_from_scene(scene_description)

            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            # Publish scene description
            scene_msg = String()
            scene_msg.data = json.dumps(scene_description, indent=2)
            self.scene_pub.publish(scene_msg)

            self.get_logger().info(
                f'Scene Understanding - Scene: {scene_description["predicted_scene"]}, '
                f'Confidence: {scene_description["scene_confidence"]:.3f}, '
                f'Objects: {len(scene_description["objects"])}, '
                f'Action - Linear: {action_cmd.linear.x:.2f}, Angular: {action_cmd.angular.z:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in scene understanding: {str(e)}')

    def generate_action_from_scene(self, scene_description):
        """Generate action based on scene understanding and command"""
        cmd = Twist()

        command_lower = scene_description['command'].lower()

        # Example action generation based on scene and command
        if 'move forward' in command_lower or 'go forward' in command_lower:
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif 'turn left' in command_lower:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif 'turn right' in command_lower:
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif 'approach' in command_lower or 'go to' in command_lower:
            # Look for relevant objects in scene
            relevant_objects = [obj for obj in scene_description['objects']
                              if obj['class'] in command_lower or
                              any(attr['attribute'] in command_lower for attr in obj.get('attributes', []))]

            if relevant_objects:
                # Move toward the first relevant object
                target_obj = relevant_objects[0]
                cmd.linear.x = 0.2
                cmd.angular.z = 0.0  # Simplified - in reality, adjust based on position
            else:
                cmd.linear.x = 0.3  # Default forward movement
                cmd.angular.z = 0.0
        elif 'avoid' in command_lower or 'stop' in command_lower:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Default action based on scene
            if scene_description['predicted_scene'] == 'office':
                cmd.linear.x = 0.2  # Move slowly in office
                cmd.angular.z = 0.0
            elif scene_description['predicted_scene'] == 'outdoor':
                cmd.linear.x = 0.4  # Move faster outdoors
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.3  # Default speed
                cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    scene_understanding = MultimodalSceneUnderstanding()

    try:
        rclpy.spin(scene_understanding)
    except KeyboardInterrupt:
        scene_understanding.get_logger().info('Multimodal scene understanding shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        scene_understanding.action_pub.publish(cmd)

        scene_understanding.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Multimodal Fusion

1. **Synchronization**: Ensure proper temporal alignment between modalities
2. **Calibration**: Maintain proper calibration between sensors and coordinate systems
3. **Uncertainty Management**: Propagate uncertainty through the fusion process
4. **Robustness**: Handle missing or corrupted modality data gracefully
5. **Efficiency**: Optimize fusion algorithms for real-time performance
6. **Interpretability**: Provide insights into how modalities contribute to decisions
7. **Validation**: Validate fusion results against ground truth when possible

### Physical Grounding and Simulation-to-Real Mapping

When implementing multimodal fusion systems:

- **Sensor Calibration**: Ensure proper calibration between different sensor modalities
- **Temporal Alignment**: Account for different sensor update rates and latencies
- **Coordinate Systems**: Maintain consistent coordinate system transformations
- **Environmental Conditions**: Consider how environmental factors affect different modalities
- **Computational Constraints**: Account for real-time processing requirements on physical hardware
- **Failure Modes**: Plan for scenarios where one or more modalities fail
- **Safety Considerations**: Implement safety checks around fused decisions

### Troubleshooting Multimodal Fusion Issues

Common multimodal fusion problems and solutions:

- **Synchronization Issues**: Check timing alignment between different sensors
- **Calibration Problems**: Verify sensor calibrations and coordinate transforms
- **Confidence Miscalibration**: Adjust confidence estimation methods
- **Performance Issues**: Optimize fusion algorithms for real-time operation
- **Integration Problems**: Ensure proper data flow between modalities
- **Uncertainty Propagation**: Verify uncertainty is properly handled in fusion

### Summary

This chapter covered multimodal fusion techniques that integrate information from multiple sensory modalities to create comprehensive understanding and coordinated action in robotic systems. You learned about different fusion architectures (early, late, intermediate), cross-modal attention mechanisms, confidence-based weighting, and scene understanding approaches. Multimodal fusion enables robots to make more informed decisions by combining complementary information from different sensors, resulting in more robust and capable robotic systems. The integration of vision, language, and other modalities allows robots to understand and interact with their environment in more human-like ways. In the next chapter, we'll explore advanced topics in Physical AI and humanoid robotics.