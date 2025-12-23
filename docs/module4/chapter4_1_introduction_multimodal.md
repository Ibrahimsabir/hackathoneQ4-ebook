# Module 4: Vision–Language–Action (VLA)

## Chapter 4.1: Introduction to Multimodal AI

This chapter introduces multimodal AI systems that integrate vision, language, and action for robotics applications. Multimodal AI enables robots to understand and interact with the world through multiple sensory channels, creating more natural and capable robotic systems.

### Understanding Multimodal AI

Multimodal AI combines multiple types of sensory input to create more comprehensive understanding and decision-making capabilities. In robotics, this typically involves:

- **Vision**: Processing visual information from cameras and sensors
- **Language**: Understanding and generating human language
- **Action**: Executing physical actions in the environment
- **Other modalities**: Touch, hearing, smell, etc.

The integration of these modalities enables robots to:
- Understand natural language commands in visual contexts
- Reason about spatial relationships between objects
- Execute complex tasks requiring both perception and planning
- Interact naturally with humans

### Multimodal AI Architecture

The architecture of multimodal AI systems typically includes:

```
+-------------------+
|   Language        |
|   Understanding   |
+-------------------+
|   Visual          |
|   Processing      |
+-------------------+
|   Action          |
|   Planning        |
+-------------------+
|   Fusion Layer    |
|   (Cross-modal    |
|   Attention)      |
+-------------------+
|   Decision Layer  |
|   (End-to-end)    |
+-------------------+
```

### Vision-Language-Action Integration

Implementing a basic VLA system:

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

class VLASystem(Node):
    def __init__(self):
        super().__init__('vla_system')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize VLA components
        self.vision_encoder = None
        self.language_encoder = None
        self.action_decoder = None
        self.cross_attention = None
        self.initialize_vla_models()

        # State variables
        self.current_image = None
        self.current_command = None
        self.current_detections = None
        self.vision_features = None
        self.language_features = None

        # Control timer
        self.vla_timer = self.create_timer(0.1, self.vla_processing_loop)

        self.get_logger().info('VLA system initialized')

    def initialize_vla_models(self):
        """Initialize VLA models (simplified)"""
        try:
            # Vision encoder: Simple CNN for image feature extraction
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc = nn.Linear(64 * 16 * 16, 256)  # Assuming 64x64 input

                def forward(self, x):
                    x = self.pool(F.relu(self.conv1(x)))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 16 * 16)
                    x = F.relu(self.fc(x))
                    return x

            # Language encoder: Simple embedding model (simplified)
            class LanguageEncoder(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=256):
                    super(LanguageEncoder, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.fc = nn.Linear(embed_dim, 256)

                def forward(self, x):
                    x = self.embedding(x)
                    x = torch.mean(x, dim=1)  # Average pooling over sequence
                    x = self.fc(x)
                    return x

            # Cross-attention module
            class CrossAttention(nn.Module):
                def __init__(self, feature_dim=256):
                    super(CrossAttention, self).__init__()
                    self.query = nn.Linear(feature_dim, feature_dim)
                    self.key = nn.Linear(feature_dim, feature_dim)
                    self.value = nn.Linear(feature_dim, feature_dim)

                def forward(self, vision_features, language_features):
                    Q = self.query(language_features)
                    K = self.key(vision_features)
                    V = self.value(vision_features)

                    attention_scores = torch.matmul(Q, K.transpose(-2, -1))
                    attention_weights = F.softmax(attention_scores, dim=-1)
                    attended_features = torch.matmul(attention_weights, V)

                    # Concatenate with original features
                    fused_features = torch.cat([attended_features, language_features], dim=-1)
                    return fused_features

            # Action decoder
            class ActionDecoder(nn.Module):
                def __init__(self, input_dim=512, output_dim=2):  # 2 for linear/angular velocities
                    super(ActionDecoder, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.output = nn.Linear(128, output_dim)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.output(x)
                    return x

            # Initialize models
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
            self.cross_attention = CrossAttention()
            self.action_decoder = ActionDecoder()

            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.cross_attention.eval()
            self.action_decoder.eval()

            self.get_logger().info('VLA models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize VLA models: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def vla_processing_loop(self):
        """Main VLA processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Process image through vision encoder
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            vision_features = self.encode_vision(cv_image)

            # Process command through language encoder
            language_features = self.encode_language(self.current_command)

            # Fuse vision and language features
            fused_features = self.cross_attention(vision_features, language_features)

            # Decode action
            action_output = self.decode_action(fused_features)

            # Execute action
            cmd = Twist()
            cmd.linear.x = float(action_output[0][0])
            cmd.angular.z = float(action_output[0][1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            self.cmd_vel_pub.publish(cmd)

            self.get_logger().info(
                f'VLA Command: "{self.current_command}", '
                f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in VLA processing: {str(e)}')

    def encode_vision(self, image):
        """Encode visual information"""
        # Preprocess image
        import cv2
        image_resized = cv2.resize(image, (64, 64))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
        image_tensor = torch.FloatTensor(image_tensor)

        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)
            return vision_features

    def encode_language(self, command):
        """Encode language command"""
        # Simple tokenization and embedding (simplified)
        # In a real system, you'd use a proper tokenizer
        tokens = command.lower().split()
        token_ids = [hash(token) % 10000 for token in tokens]  # Simplified hashing
        token_tensor = torch.LongTensor([token_ids]).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            language_features = self.language_encoder(token_tensor)
            return language_features

    def decode_action(self, fused_features):
        """Decode action from fused features"""
        with torch.no_grad():
            action_output = self.action_decoder(fused_features)
            return action_output

def main(args=None):
    rclpy.init(args=args)
    vla_system = VLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        vla_system.get_logger().info('VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        vla_system.cmd_vel_pub.publish(cmd)

        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Vision Processing for Multimodal AI

Advanced vision processing for multimodal systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image as PILImage

class VLAVisionProcessor(Node):
    def __init__(self):
        super().__init__('vla_vision_processor')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/vla_detections',
            10
        )

        self.attention_map_pub = self.create_publisher(
            Image,
            '/attention_map',
            10
        )

        # Initialize vision models
        self.feature_extractor = None
        self.attention_model = None
        self.initialize_vision_models()

        # State variables
        self.current_image = None
        self.feature_maps = None

        # Control timer
        self.vision_timer = self.create_timer(0.05, self.vision_processing_loop)

        self.get_logger().info('VLA vision processor initialized')

    def initialize_vision_models(self):
        """Initialize vision processing models (simplified)"""
        try:
            # Feature extractor: ResNet-like architecture
            class FeatureExtractor(nn.Module):
                def __init__(self):
                    super(FeatureExtractor, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

                def forward(self, x):
                    features = self.conv_layers(x)
                    features = self.adaptive_pool(features)
                    return features

            # Attention model for spatial attention
            class SpatialAttention(nn.Module):
                def __init__(self, channels=256):
                    super(SpatialAttention, self).__init__()
                    self.conv1 = nn.Conv2d(channels, channels // 8, 1)
                    self.conv2 = nn.Conv2d(channels // 8, 1, 1)

                def forward(self, x):
                    attention = self.conv1(x)
                    attention = F.relu(attention)
                    attention = self.conv2(attention)
                    attention = torch.sigmoid(attention)
                    return x * attention

            # Initialize models
            self.feature_extractor = FeatureExtractor()
            self.attention_model = SpatialAttention()
            self.feature_extractor.eval()
            self.attention_model.eval()

            self.get_logger().info('Vision models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize vision models: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def vision_processing_loop(self):
        """Main vision processing loop"""
        if (self.feature_extractor is None or
            self.current_image is None):
            return

        try:
            # Process image
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")

            # Extract features
            features = self.extract_visual_features(cv_image)
            self.feature_maps = features

            # Apply attention mechanism
            attended_features = self.attention_model(features)

            # Generate detections
            detections = self.generate_detections(attended_features, cv_image)

            # Publish detections
            if detections:
                detections.header = self.current_image.header
                self.detection_pub.publish(detections)

            # Publish attention map visualization
            attention_map = self.create_attention_visualization(attended_features)
            if attention_map is not None:
                attention_msg = self.bridge.cv2_to_imgmsg(attention_map, "bgr8")
                attention_msg.header = self.current_image.header
                self.attention_map_pub.publish(attention_msg)

            self.get_logger().info(f'Processed image: {cv_image.shape}, generated {len(detections.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Error in vision processing: {str(e)}')

    def extract_visual_features(self, image):
        """Extract visual features using CNN"""
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        pil_image = PILImage.fromarray(image)
        image_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            return features

    def generate_detections(self, features, original_image):
        """Generate object detections from features"""
        from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
        from geometry_msgs.msg import Point

        detections = Detection2DArray()
        detections.header.frame_id = 'camera_frame'

        # In a real system, this would use a proper detection head
        # For demonstration, we'll simulate detections
        height, width = original_image.shape[:2]

        # Simulate detection of prominent regions in feature map
        feature_map = features.squeeze(0).mean(dim=0)  # Average across channels
        feature_map_np = feature_map.numpy()

        # Find salient regions (simplified)
        salient_points = np.where(feature_map_np > np.mean(feature_map_np))

        for i in range(min(5, len(salient_points[0]))):  # Limit to 5 detections
            if i < len(salient_points[0]):
                y_idx = salient_points[0][i]
                x_idx = salient_points[1][i]

                # Map to original image coordinates
                orig_x = int((x_idx / features.shape[3]) * width)
                orig_y = int((y_idx / features.shape[2]) * height)

                detection = Detection2D()

                # Set bounding box (simplified)
                detection.bbox.center.x = orig_x
                detection.bbox.center.y = orig_y
                detection.bbox.size_x = 50  # pixels
                detection.bbox.size_y = 50  # pixels

                # Add classification result
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'object'
                hypothesis.hypothesis.score = float(feature_map_np[y_idx, x_idx])
                detection.results.append(hypothesis)

                detections.detections.append(detection)

        return detections

    def create_attention_visualization(self, attended_features):
        """Create visualization of attention weights"""
        import cv2

        # Get attention weights from the attended features
        # For visualization, we'll create a heatmap
        if attended_features is not None:
            # Average across channels to get spatial attention
            attention_map = attended_features.squeeze(0).mean(dim=0)

            # Normalize to 0-255 range
            attention_map = attention_map - torch.min(attention_map)
            attention_map = attention_map / torch.max(attention_map)
            attention_map = (attention_map * 255).byte()

            # Convert to numpy and create heatmap
            attention_np = attention_map.numpy()
            heatmap = cv2.applyColorMap(attention_np, cv2.COLORMAP_JET)

            return heatmap

        return None

def main(args=None):
    rclpy.init(args=args)
    vision_processor = VLAVisionProcessor()

    try:
        rclpy.spin(vision_processor)
    except KeyboardInterrupt:
        vision_processor.get_logger().info('VLA vision processor shutting down')
    finally:
        vision_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Language Understanding for VLA Systems

Implementing language understanding for multimodal robotics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class VLALanguageProcessor(Node):
    def __init__(self):
        super().__init__('vla_language_processor')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            '/parsed_intent',
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize language models
        self.intent_classifier = None
        self.entity_extractor = None
        self.initialize_language_models()

        # State variables
        self.current_command = None
        self.current_detections = None
        self.parsed_intent = None
        self.entities = {}

        # Intent-action mapping
        self.intent_action_map = {
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'approach_object': self.approach_object,
            'avoid_object': self.avoid_object,
            'stop': self.stop_robot
        }

        # Control timer
        self.language_timer = self.create_timer(0.1, self.language_processing_loop)

        self.get_logger().info('VLA language processor initialized')

    def initialize_language_models(self):
        """Initialize language processing models (simplified)"""
        try:
            # Intent classifier: Simple rule-based classifier (simplified)
            class IntentClassifier(nn.Module):
                def __init__(self, num_intents=8):
                    super(IntentClassifier, self).__init__()
                    self.num_intents = num_intents
                    self.fc1 = nn.Linear(300, 256)  # Assuming 300-dim embeddings
                    self.fc2 = nn.Linear(256, 128)
                    self.output = nn.Linear(128, num_intents)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.output(x)
                    return F.softmax(x, dim=1)

            # Entity extractor
            class EntityExtractor:
                def __init__(self):
                    self.object_keywords = ['box', 'ball', 'chair', 'table', 'person', 'robot', 'object']
                    self.color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white']
                    self.size_keywords = ['big', 'small', 'large', 'tiny', 'huge', 'little']

                def extract_entities(self, text):
                    entities = {
                        'objects': [],
                        'colors': [],
                        'sizes': [],
                        'positions': []
                    }

                    text_lower = text.lower()

                    # Extract objects
                    for keyword in self.object_keywords:
                        if keyword in text_lower:
                            entities['objects'].append(keyword)

                    # Extract colors
                    for keyword in self.color_keywords:
                        if keyword in text_lower:
                            entities['colors'].append(keyword)

                    # Extract sizes
                    for keyword in self.size_keywords:
                        if keyword in text_lower:
                            entities['sizes'].append(keyword)

                    # Extract positions/directions
                    if 'left' in text_lower:
                        entities['positions'].append('left')
                    if 'right' in text_lower:
                        entities['positions'].append('right')
                    if 'front' in text_lower or 'ahead' in text_lower:
                        entities['positions'].append('front')
                    if 'behind' in text_lower or 'back' in text_lower:
                        entities['positions'].append('back')

                    return entities

            # Initialize models
            self.intent_classifier = IntentClassifier()
            self.entity_extractor = EntityExtractor()
            self.intent_classifier.eval()

            self.get_logger().info('Language models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize language models: {str(e)}')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command: "{msg.data}"')

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def language_processing_loop(self):
        """Main language processing loop"""
        if self.current_command is None:
            return

        try:
            # Parse command
            intent = self.parse_intent(self.current_command)
            entities = self.entity_extractor.extract_entities(self.current_command)

            # Store parsed information
            self.parsed_intent = intent
            self.entities = entities

            # Execute action based on intent
            if intent in self.intent_action_map:
                action_cmd = self.intent_action_map[intent]()
                if action_cmd is not None:
                    self.action_pub.publish(action_cmd)

            # Publish parsed intent
            intent_msg = String()
            intent_msg.data = f'{intent}: {entities}'
            self.intent_pub.publish(intent_msg)

            self.get_logger().info(
                f'Parsed intent: {intent}, Entities: {entities}, '
                f'Command: "{self.current_command}"'
            )

        except Exception as e:
            self.get_logger().error(f'Error in language processing: {str(e)}')

    def parse_intent(self, command):
        """Parse intent from command (simplified)"""
        command_lower = command.lower()

        # Simple rule-based intent classification (in a real system, this would use ML)
        if any(word in command_lower for word in ['forward', 'ahead', 'go', 'move']):
            if any(word in command_lower for word in ['forward', 'ahead']):
                return 'move_forward'
            elif any(word in command_lower for word in ['backward', 'back']):
                return 'move_backward'
            else:
                return 'move_forward'  # Default forward movement

        elif any(word in command_lower for word in ['left', 'turn left', 'rotate left']):
            return 'turn_left'

        elif any(word in command_lower for word in ['right', 'turn right', 'rotate right']):
            return 'turn_right'

        elif any(word in command_lower for word in ['approach', 'go to', 'toward', 'move to']):
            return 'approach_object'

        elif any(word in command_lower for word in ['avoid', 'stay away', 'move away']):
            return 'avoid_object'

        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return 'stop'

        else:
            # Default to forward movement
            return 'move_forward'

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = 0.0
        return cmd

    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.3
        cmd.angular.z = 0.0
        return cmd

    def turn_left(self):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5
        return cmd

    def turn_right(self):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5
        return cmd

    def approach_object(self):
        """Approach detected object"""
        if self.current_detections and len(self.current_detections.detections) > 0:
            # Approach the first detected object (simplified)
            cmd = Twist()
            cmd.linear.x = 0.2  # Move forward slowly
            cmd.angular.z = 0.0
            return cmd

        # If no objects detected, move forward
        return self.move_forward()

    def avoid_object(self):
        """Avoid detected objects"""
        if self.current_detections and len(self.current_detections.detections) > 0:
            # For simplicity, turn to avoid
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3  # Turn to avoid
            return cmd

        # If no objects detected, no action needed
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def stop_robot(self):
        """Stop robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

def main(args=None):
    rclpy.init(args=args)
    lang_processor = VLALanguageProcessor()

    try:
        rclpy.spin(lang_processor)
    except KeyboardInterrupt:
        lang_processor.get_logger().info('VLA language processor shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        lang_processor.action_pub.publish(cmd)

        lang_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Cross-Modal Attention Mechanisms

Implementing cross-modal attention for vision-language fusion:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(Node):
    def __init__(self):
        super().__init__('cross_modal_attention')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize attention models
        self.vision_encoder = None
        self.language_encoder = None
        self.cross_attention = None
        self.action_decoder = None
        self.initialize_attention_models()

        # State variables
        self.current_image = None
        self.current_command = None
        self.current_detections = None

        # Attention weights for visualization
        self.attention_weights = None

        # Control timer
        self.attention_timer = self.create_timer(0.1, self.attention_processing_loop)

        self.get_logger().info('Cross-modal attention system initialized')

    def initialize_attention_models(self):
        """Initialize cross-modal attention models (simplified)"""
        try:
            # Vision encoder
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))  # 8x8 spatial grid

                def forward(self, x):
                    features = self.conv_layers(x)
                    features = self.spatial_pool(features)  # Shape: [batch, channels, 8, 8]
                    # Reshape to [batch, spatial_locations, features]
                    batch_size, channels, h, w = features.shape
                    features = features.view(batch_size, channels, h * w).transpose(1, 2)
                    return features  # [batch, 64, 128] assuming 128 feature channels

            # Language encoder
            class LanguageEncoder(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256):
                    super(LanguageEncoder, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, 256)

                def forward(self, x):
                    # x shape: [batch, seq_len]
                    embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
                    lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim]
                    # Take the last output
                    last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
                    output = self.fc(last_output)  # [batch, 256]
                    return output.unsqueeze(1)  # [batch, 1, 256] for attention

            # Cross-modal attention module
            class CrossModalAttention(nn.Module):
                def __init__(self, feature_dim=256):
                    super(CrossModalAttention, self).__init__()
                    self.vision_proj = nn.Linear(128, feature_dim)  # Vision features -> 256 dim
                    self.lang_proj = nn.Linear(256, feature_dim)    # Language features -> 256 dim
                    self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
                    self.layer_norm = nn.LayerNorm(feature_dim)

                def forward(self, vision_features, language_features):
                    # vision_features: [batch, spatial_locs, vision_dim=128]
                    # language_features: [batch, 1, lang_dim=256]

                    # Project to same dimension
                    vision_proj = self.vision_proj(vision_features)  # [batch, spatial_locs, 256]
                    lang_proj = self.lang_proj(language_features)    # [batch, 1, 256]

                    # Apply multi-head attention
                    attended_vision, attention_weights = self.attention(
                        query=lang_proj,      # [batch, 1, 256]
                        key=vision_proj,      # [batch, spatial_locs, 256]
                        value=vision_proj     # [batch, spatial_locs, 256]
                    )

                    # Residual connection and layer norm
                    attended_features = self.layer_norm(attended_vision + lang_proj)

                    return attended_features, attention_weights

            # Action decoder
            class ActionDecoder(nn.Module):
                def __init__(self, input_dim=256, output_dim=2):
                    super(ActionDecoder, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.output = nn.Linear(64, output_dim)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = F.relu(self.fc3(x))
                    x = self.output(x)
                    return x

            # Initialize models
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
            self.cross_attention = CrossModalAttention()
            self.action_decoder = ActionDecoder()

            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.cross_attention.eval()
            self.action_decoder.eval()

            self.get_logger().info('Cross-modal attention models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize attention models: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def attention_processing_loop(self):
        """Main cross-modal attention processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Encode vision features
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            vision_features = self.encode_vision(cv_image)

            # Encode language features
            language_features = self.encode_language(self.current_command)

            # Apply cross-modal attention
            attended_features, attention_weights = self.cross_attention(
                vision_features, language_features
            )

            # Store attention weights for potential visualization
            self.attention_weights = attention_weights

            # Decode action
            action_output = self.decode_action(attended_features)

            # Create and publish action command
            cmd = Twist()
            cmd.linear.x = float(action_output[0, 0, 0])
            cmd.angular.z = float(action_output[0, 0, 1])

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            self.action_pub.publish(cmd)

            self.get_logger().info(
                f'Cross-Modal Attention - Command: "{self.current_command}", '
                f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in cross-modal attention: {str(e)}')

    def encode_vision(self, image):
        """Encode visual features"""
        import cv2
        # Preprocess image
        image_resized = cv2.resize(image, (64, 64))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
        image_tensor = torch.FloatTensor(image_tensor)

        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)
            return vision_features

    def encode_language(self, command):
        """Encode language command"""
        # Simple tokenization (in a real system, use proper tokenizer)
        tokens = command.lower().split()
        # Convert to token IDs using simple hashing
        token_ids = [hash(token) % 10000 for token in tokens]

        # Pad or truncate to fixed length
        max_length = 20
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))  # Padding
        else:
            token_ids = token_ids[:max_length]  # Truncate

        token_tensor = torch.LongTensor([token_ids])  # [1, max_length]

        with torch.no_grad():
            language_features = self.language_encoder(token_tensor)
            return language_features

    def decode_action(self, attended_features):
        """Decode action from attended features"""
        with torch.no_grad():
            action_output = self.action_decoder(attended_features)
            return action_output

def main(args=None):
    rclpy.init(args=args)
    attention_system = CrossModalAttention()

    try:
        rclpy.spin(attention_system)
    except KeyboardInterrupt:
        attention_system.get_logger().info('Cross-modal attention system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        attention_system.action_pub.publish(cmd)

        attention_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### VLA System Integration

Bringing together vision, language, and action components:

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

class IntegratedVLASystem(Node):
    def __init__(self):
        super().__init__('integrated_vla_system')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize integrated models
        self.vla_model = None
        self.initialize_integrated_model()

        # State variables
        self.current_image = None
        self.current_command = None
        self.current_detections = None
        self.system_state = 'idle'

        # Control timer
        self.vla_timer = self.create_timer(0.05, self.integrated_vla_loop)

        self.get_logger().info('Integrated VLA system initialized')

    def initialize_integrated_model(self):
        """Initialize integrated VLA model (simplified)"""
        try:
            # Integrated VLA model
            class IntegratedVLAModel(nn.Module):
                def __init__(self):
                    super(IntegratedVLAModel, self).__init__()

                    # Vision encoder
                    self.vision_conv = nn.Sequential(
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
                    self.lang_embedding = nn.Embedding(10000, 128)
                    self.lang_lstm = nn.LSTM(128, 256, batch_first=True)

                    # Cross-modal fusion
                    self.vision_fc = nn.Linear(128 * 8 * 8, 512)
                    self.lang_fc = nn.Linear(256, 512)
                    self.fusion = nn.Linear(512 + 512, 512)

                    # Action decoder
                    self.action_head = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # linear and angular velocities
                    )

                def forward(self, image, language_tokens):
                    # Process vision
                    vision_features = self.vision_conv(image)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.relu(self.vision_fc(vision_features))

                    # Process language
                    lang_embedded = self.lang_embedding(language_tokens)
                    lang_output, _ = self.lang_lstm(lang_embedded)
                    # Use last output
                    lang_features = lang_output[:, -1, :]
                    lang_features = F.relu(self.lang_fc(lang_features))

                    # Fuse modalities
                    fused_features = torch.cat([vision_features, lang_features], dim=1)
                    fused_features = F.relu(self.fusion(fused_features))

                    # Generate action
                    action = self.action_head(fused_features)
                    return action

            # Initialize model
            self.vla_model = IntegratedVLAModel()
            self.vla_model.eval()

            self.get_logger().info('Integrated VLA model initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize integrated model: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def integrated_vla_loop(self):
        """Main integrated VLA processing loop"""
        if (self.vla_model is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Prepare inputs
            image_tensor = self.prepare_image_input(self.current_image)
            language_tensor = self.prepare_language_input(self.current_command)

            # Run integrated VLA model
            with torch.no_grad():
                action_output = self.vla_model(image_tensor, language_tensor)

            # Extract action
            linear_vel = float(action_output[0, 0])
            angular_vel = float(action_output[0, 1])

            # Create and publish action command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            self.action_pub.publish(cmd)

            self.get_logger().info(
                f'Integrated VLA - Command: "{self.current_command}", '
                f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
            )

            # Update system state
            if abs(cmd.linear.x) > 0.01 or abs(cmd.angular.z) > 0.01:
                self.system_state = 'executing'
            else:
                self.system_state = 'waiting'

        except Exception as e:
            self.get_logger().error(f'Error in integrated VLA: {str(e)}')

    def prepare_image_input(self, image_msg):
        """Prepare image for model input"""
        import cv2
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Preprocess image
        image_resized = cv2.resize(cv_image, (64, 64))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
        image_tensor = torch.FloatTensor(image_tensor)

        return image_tensor

    def prepare_language_input(self, command):
        """Prepare language command for model input"""
        # Tokenize command
        tokens = command.lower().split()
        token_ids = [hash(token) % 10000 for token in tokens]

        # Pad to fixed length
        max_length = 20
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))
        else:
            token_ids = token_ids[:max_length]

        token_tensor = torch.LongTensor([token_ids])  # [1, max_length]

        return token_tensor

def main(args=None):
    rclpy.init(args=args)
    integrated_system = IntegratedVLASystem()

    try:
        rclpy.spin(integrated_system)
    except KeyboardInterrupt:
        integrated_system.get_logger().info('Integrated VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        integrated_system.action_pub.publish(cmd)

        integrated_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Multimodal AI

1. **Modality Alignment**: Ensure proper alignment between different modalities
2. **Feature Normalization**: Normalize features across different modalities
3. **Attention Mechanisms**: Use attention to focus on relevant information
4. **Robustness**: Handle missing or noisy modality inputs gracefully
5. **Latency Considerations**: Optimize for real-time performance
6. **Calibration**: Calibrate sensors and models for accurate fusion
7. **Evaluation**: Use appropriate metrics for multimodal systems

### Physical Grounding and Simulation-to-Real Mapping

When implementing multimodal AI systems:

- **Sensor Calibration**: Ensure accurate calibration of vision and other sensors
- **Latency Compensation**: Account for processing delays in real-time systems
- **Environmental Conditions**: Consider lighting, noise, and other environmental factors
- **Hardware Constraints**: Account for computational limitations on real robots
- **Safety Systems**: Implement safety checks around multimodal decision-making
- **Validation**: Regularly validate system behavior in real-world conditions

### Troubleshooting Multimodal AI Issues

Common multimodal AI problems and solutions:

- **Modality Misalignment**: Check timing synchronization between sensors
- **Feature Dimension Mismatch**: Verify feature dimensions match model expectations
- **Attention Failure**: Inspect attention weights for proper focus
- **Performance Issues**: Profile each modality separately
- **Integration Problems**: Test components individually before integration

### Summary

This chapter introduced multimodal AI systems that integrate vision, language, and action for robotics applications. You learned about the architecture of VLA systems, how to implement vision and language processing components, cross-modal attention mechanisms, and how to integrate these components into a cohesive system. Multimodal AI enables robots to understand and interact with the world through multiple sensory channels, creating more natural and capable robotic systems. In the next chapter, we'll explore vision processing in more detail for multimodal robotics applications.