# Module 4: Vision–Language–Action (VLA)

## Chapter 4.8: VLA Integration Patterns and Best Practices

This chapter focuses on integration patterns and best practices for implementing Vision-Language-Action (VLA) systems in robotics. We'll explore how to effectively combine vision, language, and action components into cohesive robotic systems that demonstrate intelligent behavior.

### Understanding VLA Integration Patterns

VLA integration involves connecting vision processing, language understanding, and action execution in a way that enables coherent robotic behavior. Key integration patterns include:

- **Sequential Integration**: Vision → Language → Action pipeline
- **Parallel Integration**: All modalities processed simultaneously
- **Hierarchical Integration**: Multi-level processing with high-level planning and low-level execution
- **Cross-Modal Attention**: Direct interaction between modalities during processing
- **Memory-Augmented Integration**: Explicit memory for long-term reasoning

### VLA Integration Architecture

The typical VLA integration architecture follows this pattern:

```
+-------------------+    +-------------------+    +-------------------+
|   Vision          |    |   Language        |    |   Action          |
|   Processing      |    |   Understanding   |    |   Execution       |
|   (Images,        |    |   (Commands,      |    |   (Movement,      |
|    Objects)       |    |    Intent)        |    |    Manipulation)  |
+-------------------+    +-------------------+    +-------------------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------+
|                 Cross-Modal Attention & Fusion             |
|                (Joint Representation)                     |
+------------------------------------------------------------+
         |                        |
         v                        v
+-------------------+    +-------------------+
|   Decision        |    |   Execution       |
|   Making          |    |   Coordination    |
|   (Planning,      |    |   (Motor Control,|
|    Reasoning)     |    |    Safety)       |
+-------------------+    +-------------------+
```

### Sequential VLA Integration Pattern

Implementing the sequential integration pattern where each modality is processed in sequence:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json

class SequentialVLANode(Node):
    def __init__(self):
        super().__init__('sequential_vla')

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

        self.sequential_status_pub = self.create_publisher(
            String,
            '/sequential_vla_status',
            10
        )

        # Initialize sequential VLA components
        self.vision_processor = None
        self.language_processor = None
        self.action_generator = None
        self.initialize_sequential_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.vision_output = None
        self.language_output = None

        # Control timer
        self.sequential_timer = self.create_timer(0.1, self.sequential_vla_loop)

        self.get_logger().info('Sequential VLA system initialized')

    def initialize_sequential_components(self):
        """Initialize sequential VLA processing components"""
        try:
            # Vision processing module
            class VisionProcessor(nn.Module):
                def __init__(self):
                    super(VisionProcessor, self).__init__()

                    # Feature extraction
                    self.features = nn.Sequential(
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

                    # Object detection head
                    self.obj_detection = nn.Sequential(
                        nn.Linear(256 * 8 * 8, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 4 * 80)  # 4 bbox coords * 80 classes
                    )

                    # Scene classification head
                    self.scene_classifier = nn.Sequential(
                        nn.Linear(256 * 8 * 8, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)  # 10 scene types
                    )

                def forward(self, x):
                    features = self.features(x)
                    features_flat = features.view(features.size(0), -1)

                    obj_detections = self.obj_detection(features_flat)
                    obj_detections = obj_detections.view(obj_detections.size(0), 80, 4)

                    scene_probs = self.scene_classifier(features_flat)

                    return obj_detections, scene_probs, features_flat

            # Language processing module
            class LanguageProcessor(nn.Module):
                def __init__(self):
                    super(LanguageProcessor, self).__init__()

                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True)
                    self.intent_classifier = nn.Linear(256, 10)  # 10 intent classes
                    self.entity_extractor = nn.Linear(256, 128)  # Entity features

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    last_output = lstm_out[:, -1, :]

                    intent_probs = self.intent_classifier(last_output)
                    entity_features = self.entity_extractor(last_output)

                    return intent_probs, entity_features

            # Action generation module
            class ActionGenerator(nn.Module):
                def __init__(self):
                    super(ActionGenerator, self).__init__()

                    # Fusion of vision and language features
                    self.fusion = nn.Sequential(
                        nn.Linear(256 * 8 * 8 + 256 + 128, 512),  # Vision + Intent + Entities
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                def forward(self, vision_features, intent_features, entity_features):
                    combined = torch.cat([vision_features, intent_features, entity_features], dim=1)
                    action_output = self.fusion(combined)
                    return action_output

            # Initialize models
            self.vision_processor = VisionProcessor()
            self.language_processor = LanguageProcessor()
            self.action_generator = ActionGenerator()

            # Set to evaluation mode
            self.vision_processor.eval()
            self.language_processor.eval()
            self.action_generator.eval()

            self.get_logger().info('Sequential VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize sequential VLA components: {str(e)}')

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

    def sequential_vla_loop(self):
        """Main sequential VLA processing loop"""
        if (self.vision_processor is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Step 1: Process vision
            vision_features, obj_detections, scene_probs = self.process_vision_sequentially()

            # Step 2: Process language
            intent_probs, entity_features = self.process_language_sequentially()

            # Step 3: Generate action
            action_output = self.generate_action_sequentially(
                vision_features, intent_probs, entity_features
            )

            if action_output is not None:
                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)
                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Publish status
                status_msg = String()
                status_msg.data = json.dumps({
                    'vision_processing': {
                        'detected_objects': len(obj_detections[0]) if obj_detections is not None else 0,
                        'scene_type': torch.argmax(scene_probs[0]).item() if scene_probs is not None else -1
                    },
                    'language_processing': {
                        'intent': torch.argmax(intent_probs[0]).item() if intent_probs is not None else -1,
                        'command': self.current_command
                    },
                    'action_generation': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.sequential_status_pub.publish(status_msg)

                self.get_logger().info(
                    f'Sequential VLA - Vision: {obj_detections[0].shape if obj_detections is not None else "None"}, '
                    f'Language Intent: {torch.argmax(intent_probs[0]).item() if intent_probs is not None else -1}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in sequential VLA processing: {str(e)}')

    def process_vision_sequentially(self):
        """Process vision input sequentially"""
        try:
            # Convert ROS image to tensor
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                obj_detections, scene_probs, vision_features = self.vision_processor(image_tensor)
                return vision_features, obj_detections, scene_probs

        except Exception as e:
            self.get_logger().error(f'Error in sequential vision processing: {str(e)}')
            return None, None, None

    def process_language_sequentially(self):
        """Process language input sequentially"""
        try:
            # Tokenize command
            tokens = self.current_command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            with torch.no_grad():
                intent_probs, entity_features = self.language_processor(token_tensor)
                return intent_probs, entity_features

        except Exception as e:
            self.get_logger().error(f'Error in sequential language processing: {str(e)}')
            return None, None

    def generate_action_sequentially(self, vision_features, intent_probs, entity_features):
        """Generate action based on processed vision and language"""
        try:
            if vision_features is None or intent_probs is None or entity_features is None:
                return None

            with torch.no_grad():
                action_output = self.action_generator(vision_features, intent_probs, entity_features)
                return action_output

        except Exception as e:
            self.get_logger().error(f'Error in sequential action generation: {str(e)}')
            return None

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

def main(args=None):
    rclpy.init(args=args)
    seq_vla = SequentialVLANode()

    try:
        rclpy.spin(seq_vla)
    except KeyboardInterrupt:
        seq_vla.get_logger().info('Sequential VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        seq_vla.action_pub.publish(cmd)

        seq_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parallel VLA Integration Pattern

Implementing parallel integration where all modalities are processed simultaneously:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time

class ParallelVLANode(Node):
    def __init__(self):
        super().__init__('parallel_vla')

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

        self.parallel_status_pub = self.create_publisher(
            String,
            '/parallel_vla_status',
            10
        )

        # Initialize parallel VLA components
        self.vision_processor = None
        self.language_processor = None
        self.lidar_processor = None
        self.parallel_fusion = None
        self.initialize_parallel_components()

        # Queues for parallel processing
        self.vision_queue = queue.Queue(maxsize=10)
        self.language_queue = queue.Queue(maxsize=10)
        self.lidar_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        # Processing threads
        self.vision_thread = None
        self.language_thread = None
        self.lidar_thread = None
        self.fusion_thread = None

        # State variables
        self.current_image = None
        self.current_command = None
        self.current_scan = None

        # Control timer
        self.parallel_timer = self.create_timer(0.1, self.parallel_vla_control)

        self.get_logger().info('Parallel VLA system initialized')

    def initialize_parallel_components(self):
        """Initialize parallel VLA processing components"""
        try:
            # Vision processing module
            class VisionProcessor(nn.Module):
                def __init__(self):
                    super(VisionProcessor, self).__init__()

                    self.features = nn.Sequential(
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

                    self.output = nn.Linear(256 * 8 * 8, 256)

                def forward(self, x):
                    features = self.features(x)
                    features = features.view(features.size(0), -1)
                    output = self.output(features)
                    return output

            # Language processing module
            class LanguageProcessor(nn.Module):
                def __init__(self):
                    super(LanguageProcessor, self).__init__()

                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True)
                    self.output = nn.Linear(256, 256)

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    last_output = lstm_out[:, -1, :]
                    output = self.output(last_output)
                    return output

            # LiDAR processing module
            class LidarProcessor(nn.Module):
                def __init__(self):
                    super(LidarProcessor, self).__init__()

                    self.processing = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                def forward(self, x):
                    output = self.processing(x)
                    return output

            # Parallel fusion module
            class ParallelFusion(nn.Module):
                def __init__(self):
                    super(ParallelFusion, self).__init__()

                    self.fusion = nn.Sequential(
                        nn.Linear(256 * 3, 512),  # Vision + Language + LiDAR
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                def forward(self, vision_features, language_features, lidar_features):
                    combined = torch.cat([vision_features, language_features, lidar_features], dim=1)
                    output = self.fusion(combined)
                    return output

            # Initialize models
            self.vision_processor = VisionProcessor()
            self.language_processor = LanguageProcessor()
            self.lidar_processor = LidarProcessor()
            self.parallel_fusion = ParallelFusion()

            # Set to evaluation mode
            self.vision_processor.eval()
            self.language_processor.eval()
            self.lidar_processor.eval()
            self.parallel_fusion.eval()

            # Start processing threads
            self.start_parallel_processing_threads()

            self.get_logger().info('Parallel VLA components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize parallel VLA components: {str(e)}')

    def start_parallel_processing_threads(self):
        """Start parallel processing threads"""
        try:
            self.vision_thread = threading.Thread(target=self.vision_processing_worker, daemon=True)
            self.language_thread = threading.Thread(target=self.language_processing_worker, daemon=True)
            self.lidar_thread = threading.Thread(target=self.lidar_processing_worker, daemon=True)
            self.fusion_thread = threading.Thread(target=self.fusion_processing_worker, daemon=True)

            self.vision_thread.start()
            self.language_thread.start()
            self.lidar_thread.start()
            self.fusion_thread.start()

            self.get_logger().info('Parallel processing threads started')
        except Exception as e:
            self.get_logger().error(f'Failed to start parallel processing threads: {str(e)}')

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

    def parallel_vla_control(self):
        """Control loop for parallel VLA system"""
        # Push new data to processing queues
        if self.current_image is not None:
            try:
                self.vision_queue.put_nowait({
                    'image': self.current_image,
                    'timestamp': time.time()
                })
            except queue.Full:
                pass  # Drop frame if queue is full

        if self.current_command is not None:
            try:
                self.language_queue.put_nowait({
                    'command': self.current_command,
                    'timestamp': time.time()
                })
            except queue.Full:
                pass  # Drop if queue is full

        if self.current_scan is not None:
            try:
                self.lidar_queue.put_nowait({
                    'scan': self.current_scan,
                    'timestamp': time.time()
                })
            except queue.Full:
                pass  # Drop if queue is full

        # Check for results from fusion
        try:
            result = self.result_queue.get_nowait()
            action_cmd = result['action']
            timestamp = result['timestamp']

            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

                # Publish status
                status_msg = String()
                status_msg.data = json.dumps({
                    'processing_time': time.time() - timestamp,
                    'action': {
                        'linear_x': float(action_cmd.linear.x),
                        'angular_z': float(action_cmd.angular.z)
                    }
                })
                self.parallel_status_pub.publish(status_msg)

                self.get_logger().info(
                    f'Parallel VLA - Processing Time: {time.time() - timestamp:.3f}s, '
                    f'Action - Linear: {action_cmd.linear.x:.2f}, Angular: {action_cmd.angular.z:.2f}'
                )

        except queue.Empty:
            pass  # No results available yet

    def vision_processing_worker(self):
        """Worker thread for vision processing"""
        while True:
            try:
                item = self.vision_queue.get(timeout=1.0)
                image_msg = item['image']

                # Process image
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
                image_resized = cv2.resize(cv_image, (224, 224))
                image_normalized = image_resized.astype(np.float32) / 255.0
                image_tensor = np.transpose(image_normalized, (2, 0, 1))
                image_tensor = np.expand_dims(image_tensor, axis=0)
                image_tensor = torch.FloatTensor(image_tensor)

                with torch.no_grad():
                    vision_features = self.vision_processor(image_tensor)

                # Store in shared state
                self.vision_features = vision_features

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in vision processing worker: {str(e)}')

    def language_processing_worker(self):
        """Worker thread for language processing"""
        while True:
            try:
                item = self.language_queue.get(timeout=1.0)
                command = item['command']

                # Process command
                tokens = command.lower().split()
                token_ids = [hash(token) % 10000 for token in tokens]

                max_length = 20
                if len(token_ids) < max_length:
                    token_ids.extend([0] * (max_length - len(token_ids)))
                else:
                    token_ids = token_ids[:max_length]

                token_tensor = torch.LongTensor([token_ids])

                with torch.no_grad():
                    language_features = self.language_processor(token_tensor)

                # Store in shared state
                self.language_features = language_features

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in language processing worker: {str(e)}')

    def lidar_processing_worker(self):
        """Worker thread for LiDAR processing"""
        while True:
            try:
                item = self.lidar_queue.get(timeout=1.0)
                scan_msg = item['scan']

                # Process scan
                scan_data = np.array(scan_msg.ranges)
                scan_data = np.nan_to_num(scan_data, nan=3.0)
                scan_data = np.clip(scan_data, 0.0, 3.0)

                scan_tensor = torch.FloatTensor([scan_data])

                with torch.no_grad():
                    lidar_features = self.lidar_processor(scan_tensor)

                # Store in shared state
                self.lidar_features = lidar_features

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in LiDAR processing worker: {str(e)}')

    def fusion_processing_worker(self):
        """Worker thread for fusion processing"""
        while True:
            # Check if all modalities have features available
            if (hasattr(self, 'vision_features') and
                hasattr(self, 'language_features') and
                hasattr(self, 'lidar_features')):

                try:
                    with torch.no_grad():
                        action_output = self.parallel_fusion(
                            self.vision_features,
                            self.language_features,
                            self.lidar_features
                        )

                    # Convert to command
                    cmd = Twist()
                    cmd.linear.x = float(action_output[0, 0])
                    cmd.angular.z = float(action_output[0, 1])

                    # Limit velocities
                    cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
                    cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

                    # Put result in output queue
                    result_item = {
                        'action': cmd,
                        'timestamp': time.time()
                    }

                    try:
                        self.result_queue.put_nowait(result_item)
                    except queue.Full:
                        pass  # Drop result if queue is full

                except Exception as e:
                    self.get_logger().error(f'Error in fusion processing worker: {str(e)}')

            time.sleep(0.01)  # Small sleep to prevent busy waiting

def main(args=None):
    rclpy.init(args=args)
    parallel_vla = ParallelVLANode()

    try:
        rclpy.spin(parallel_vla)
    except KeyboardInterrupt:
        parallel_vla.get_logger().info('Parallel VLA system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        parallel_vla.action_pub.publish(cmd)

        parallel_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Cross-Modal Attention Integration Pattern

Implementing cross-modal attention for direct interaction between modalities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class CrossModalAttentionNode(Node):
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

        self.attention_map_pub = self.create_publisher(
            String,
            '/cross_modal_attention_maps',
            10
        )

        # Initialize cross-modal attention components
        self.vision_encoder = None
        self.language_encoder = None
        self.lidar_encoder = None
        self.cross_attention_fusion = None
        self.initialize_cross_modal_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None

        # Control timer
        self.attention_timer = self.create_timer(0.1, self.cross_modal_attention_loop)

        self.get_logger().info('Cross-modal attention VLA system initialized')

    def initialize_cross_modal_components(self):
        """Initialize cross-modal attention components"""
        try:
            # Vision encoder with attention
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()

                    # Feature extraction
                    self.features = nn.Sequential(
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

                    # Spatial attention
                    self.spatial_attention = nn.Sequential(
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 1, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    features = self.features(x)

                    # Apply spatial attention
                    attention_weights = self.spatial_attention(features)
                    attended_features = features * attention_weights

                    return attended_features, attention_weights

            # Language encoder with attention
            class LanguageEncoder(nn.Module):
                def __init__(self):
                    super(LanguageEncoder, self).__init__()

                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)

                    # Attention over sequence
                    self.attention = nn.Linear(512, 1)  # 512 = 256*2 for bidirectional
                    self.output = nn.Linear(512, 256)  # Reduce bidirectional to unidirectional

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)

                    # Apply attention over sequence
                    attention_scores = self.attention(lstm_out)
                    attention_weights = F.softmax(attention_scores, dim=1)
                    attended_output = torch.sum(lstm_out * attention_weights, dim=1)

                    output = self.output(attended_output)
                    return output, attention_weights

            # LiDAR encoder with attention
            class LidarEncoder(nn.Module):
                def __init__(self):
                    super(LidarEncoder, self).__init__()

                    self.processing = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    # Sector attention
                    self.attention = nn.Sequential(
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Softmax(dim=1)  # Attention over sectors
                    )

                def forward(self, x):
                    features = self.processing(x)

                    # Apply attention to select important sectors
                    attention_weights = self.attention(features.unsqueeze(1))
                    attended_features = features * attention_weights.squeeze(1)

                    return attended_features, attention_weights

            # Cross-modal attention fusion
            class CrossModalAttention(nn.Module):
                def __init__(self):
                    super(CrossModalAttention, self).__init__()

                    # Multi-head attention for cross-modal fusion
                    self.vision_language_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    self.vision_lidar_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    self.language_lidar_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    # Final fusion
                    self.fusion = nn.Sequential(
                        nn.Linear(256 * 3, 512),  # Combined features
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                def forward(self, vision_features, language_features, lidar_features):
                    # Reshape for attention (add sequence dimension)
                    vision_seq = vision_features.unsqueeze(1)  # [batch, 1, 256]
                    language_seq = language_features.unsqueeze(1)  # [batch, 1, 256]
                    lidar_seq = lidar_features.unsqueeze(1)  # [batch, 1, 256]

                    # Cross-modal attention
                    # Vision-Language interaction
                    vl_attended, vl_attention_weights = self.vision_language_attention(
                        query=language_seq,
                        key=vision_seq,
                        value=vision_seq
                    )

                    # Vision-LiDAR interaction
                    vlidar_attended, vlidar_attention_weights = self.vision_lidar_attention(
                        query=vision_seq,
                        key=lidar_seq,
                        value=lidar_seq
                    )

                    # Language-LiDAR interaction
                    llidar_attended, llidar_attention_weights = self.language_lidar_attention(
                        query=language_seq,
                        key=lidar_seq,
                        value=lidar_seq
                    )

                    # Combine all attended features
                    combined = torch.cat([
                        vl_attended.squeeze(1),      # Vision-Language
                        vlidar_attended.squeeze(1),  # Vision-LiDAR
                        llidar_attended.squeeze(1)   # Language-LiDAR
                    ], dim=1)

                    # Final fusion
                    output = self.fusion(combined)
                    return output, {
                        'vl_attention': vl_attention_weights,
                        'vlidar_attention': vlidar_attention_weights,
                        'llidar_attention': llidar_attention_weights
                    }

            # Initialize models
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
            self.lidar_encoder = LidarEncoder()
            self.cross_attention_fusion = CrossModalAttention()

            # Set to evaluation mode
            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.lidar_encoder.eval()
            self.cross_attention_fusion.eval()

            self.get_logger().info('Cross-modal attention components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize cross-modal attention components: {str(e)}')

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

    def cross_modal_attention_loop(self):
        """Main cross-modal attention processing loop"""
        if (self.vision_encoder is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Extract features from all modalities with attention
            vision_features, vision_attention = self.extract_vision_features_with_attention(self.current_image)
            language_features, language_attention = self.extract_language_features_with_attention(self.current_command)
            lidar_features, lidar_attention = self.extract_lidar_features_with_attention(self.current_scan)

            if all(feat is not None for feat in [vision_features, language_features, lidar_features]):
                # Perform cross-modal attention fusion
                action_output, attention_maps = self.cross_attention_fusion(
                    vision_features, language_features, lidar_features
                )

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Publish attention maps for analysis
                attention_msg = String()
                attention_msg.data = json.dumps({
                    'vision_attention_shape': vision_attention.shape if vision_attention is not None else None,
                    'language_attention_shape': language_attention.shape if language_attention is not None else None,
                    'lidar_attention_shape': lidar_attention.shape if lidar_attention is not None else None,
                    'cross_attention_maps': {
                        'vl_attention_mean': float(torch.mean(attention_maps['vl_attention'])) if attention_maps['vl_attention'] is not None else 0.0,
                        'vlidar_attention_mean': float(torch.mean(attention_maps['vlidar_attention'])) if attention_maps['vlidar_attention'] is not None else 0.0,
                        'llidar_attention_mean': float(torch.mean(attention_maps['llidar_attention'])) if attention_maps['llidar_attention'] is not None else 0.0
                    },
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.attention_map_pub.publish(attention_msg)

                self.get_logger().info(
                    f'Cross-Modal Attention - Vision: {vision_features.shape if vision_features is not None else "None"}, '
                    f'Language: {language_features.shape if language_features is not None else "None"}, '
                    f'LiDAR: {lidar_features.shape if lidar_features is not None else "None"}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in cross-modal attention processing: {str(e)}')

    def extract_vision_features_with_attention(self, image_msg):
        """Extract vision features with spatial attention"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features, attention = self.vision_encoder(image_tensor)
                # Average spatial features for single vector
                batch_size, channels, height, width = features.shape
                features = features.view(batch_size, channels, -1)
                features = torch.mean(features, dim=2)  # Average across spatial dimensions
                return features, attention

        except Exception as e:
            self.get_logger().error(f'Error in vision attention processing: {str(e)}')
            return None, None

    def extract_language_features_with_attention(self, command):
        """Extract language features with sequence attention"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            with torch.no_grad():
                features, attention = self.language_encoder(token_tensor)
                return features, attention

        except Exception as e:
            self.get_logger().error(f'Error in language attention processing: {str(e)}')
            return None, None

    def extract_lidar_features_with_attention(self, scan_msg):
        """Extract LiDAR features with sector attention"""
        try:
            scan_data = np.array(scan_msg.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)

            scan_tensor = torch.FloatTensor([scan_data])

            with torch.no_grad():
                features, attention = self.lidar_encoder(scan_tensor)
                return features, attention

        except Exception as e:
            self.get_logger().error(f'Error in LiDAR attention processing: {str(e)}')
            return None, None

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

def main(args=None):
    rclpy.init(args=args)
    cross_modal = CrossModalAttentionNode()

    try:
        rclpy.spin(cross_modal)
    except KeyboardInterrupt:
        cross_modal.get_logger().info('Cross-modal attention VLA shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        cross_modal.action_pub.publish(cmd)

        cross_modal.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Memory-Augmented VLA Integration

Implementing VLA with explicit memory for long-term reasoning:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
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
            '/memory_augmented_status',
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
        self.procedural_memory = {}  # Learned behaviors

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None

        # Memory parameters
        self.max_memory_size = 1000
        self.memory_decay_rate = 0.99
        self.relevance_threshold = 0.3

        # Control timer
        self.memory_timer = self.create_timer(0.1, self.memory_augmented_vla_loop)

        self.get_logger().info('Memory-augmented VLA system initialized')

    def initialize_memory_components(self):
        """Initialize memory-augmented VLA components"""
        try:
            # Vision encoder
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super(VisionEncoder, self).__init__()

                    self.features = nn.Sequential(
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

                    self.output = nn.Linear(256 * 8 * 8, 256)

                def forward(self, x):
                    features = self.features(x)
                    features = features.view(features.size(0), -1)
                    output = self.output(features)
                    return output

            # Language encoder
            class LanguageEncoder(nn.Module):
                def __init__(self):
                    super(LanguageEncoder, self).__init__()

                    self.embedding = nn.Embedding(10000, 256)
                    self.lstm = nn.LSTM(256, 256, batch_first=True)
                    self.output = nn.Linear(256, 256)

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    last_output = lstm_out[:, -1, :]
                    output = self.output(last_output)
                    return output

            # Memory network with attention
            class MemoryNetwork(nn.Module):
                def __init__(self):
                    super(MemoryNetwork, self).__init__()

                    # Memory reading network
                    self.read_attention = nn.MultiheadAttention(
                        embed_dim=256,
                        num_heads=8,
                        batch_first=True
                    )

                    # Memory writing network
                    self.write_network = nn.Sequential(
                        nn.Linear(256 * 2, 256),  # Current + Memory
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                def forward(self, current_features, memory_features, read_mode=True):
                    if memory_features is not None and memory_features.size(0) > 0:
                        if read_mode:
                            # Read from memory using attention
                            attended_features, attention_weights = self.read_attention(
                                query=current_features.unsqueeze(1),
                                key=memory_features,
                                value=memory_features
                            )
                            return attended_features.squeeze(1), attention_weights
                        else:
                            # Write to memory
                            combined = torch.cat([current_features, memory_features.mean(dim=0, keepdim=True)], dim=1)
                            new_memory_entry = self.write_network(combined)
                            return new_memory_entry, None
                    else:
                        # No memory available, return current features
                        return current_features, None

            # VLA policy with memory
            class MemoryAugmentedVLAPolicy(nn.Module):
                def __init__(self):
                    super(MemoryAugmentedVLAPolicy, self).__init__()

                    # Fusion of current and memory features
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
                    combined = torch.cat([vision_features, language_features, memory_features], dim=1)
                    output = self.fusion(combined)
                    return output

            # Initialize models
            self.vision_encoder = VisionEncoder()
            self.language_encoder = LanguageEncoder()
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
            # Extract current features
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.extract_language_features(self.current_command)
            lidar_features = self.extract_lidar_features(self.current_scan) if self.current_scan else torch.zeros(1, 256)

            if all(feat is not None for feat in [vision_features, language_features]):
                # Retrieve relevant memories
                relevant_memory = self.retrieve_relevant_memories(
                    vision_features, language_features, lidar_features
                )

                # Fuse current features with memory
                with torch.no_grad():
                    if relevant_memory is not None:
                        attended_memory, attention_weights = self.memory_network(
                            current_features=vision_features,
                            memory_features=relevant_memory,
                            read_mode=True
                        )
                    else:
                        attended_memory = vision_features
                        attention_weights = None

                    # Generate action using current + memory features
                    action_output = self.vla_policy(
                        vision_features, language_features, attended_memory
                    )

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Update memory with current experience
                self.update_memory(
                    vision_features, language_features, cmd,
                    self.current_command, self.current_detections
                )

                # Publish memory status
                memory_msg = String()
                memory_msg.data = json.dumps({
                    'episodic_memory_size': len(self.episodic_memory),
                    'semantic_memory_size': len(self.semantic_memory),
                    'retrieved_memory_size': relevant_memory.shape[0] if relevant_memory is not None else 0,
                    'attention_weights_mean': float(torch.mean(attention_weights)) if attention_weights is not None else 0.0,
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.memory_status_pub.publish(memory_msg)

                self.get_logger().info(
                    f'Memory-Augmented VLA - Episodic Mem: {len(self.episodic_memory)}, '
                    f'Semantic Mem: {len(self.semantic_memory)}, '
                    f'Retrieved: {relevant_memory.shape[0] if relevant_memory is not None else 0}, '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in memory-augmented VLA processing: {str(e)}')

    def retrieve_relevant_memories(self, vision_features, language_features, lidar_features):
        """Retrieve relevant memories based on current context"""
        if not self.episodic_memory:
            return None

        try:
            # Calculate relevance scores for all memories
            current_context = torch.cat([vision_features, language_features, lidar_features], dim=1)

            relevant_memories = []
            for memory_entry in self.episodic_memory[-50:]:  # Check recent memories
                memory_features = torch.cat([
                    torch.FloatTensor(memory_entry['vision_features']).unsqueeze(0),
                    torch.FloatTensor(memory_entry['language_features']).unsqueeze(0),
                    torch.FloatTensor(memory_entry['lidar_features']).unsqueeze(0)
                ], dim=1)

                # Calculate cosine similarity
                similarity = F.cosine_similarity(current_context, memory_features, dim=1)

                if similarity.item() > self.relevance_threshold:
                    relevant_memories.append(
                        torch.FloatTensor(memory_entry['memory_features']).unsqueeze(0)
                    )

            if relevant_memories:
                return torch.cat(relevant_memories, dim=0)
            else:
                return None

        except Exception as e:
            self.get_logger().error(f'Error retrieving memories: {str(e)}')
            return None

    def update_memory(self, vision_features, language_features, action_cmd, command, detections):
        """Update memory with current experience"""
        try:
            # Create memory entry
            memory_entry = {
                'timestamp': datetime.now(),
                'vision_features': vision_features.detach().cpu().numpy(),
                'language_features': language_features.detach().cpu().numpy(),
                'lidar_features': self.extract_lidar_features(self.current_scan).detach().cpu().numpy() if self.current_scan else np.zeros(256),
                'action': [action_cmd.linear.x, action_cmd.angular.z] if action_cmd else [0.0, 0.0],
                'command': command,
                'detections': self.format_detections(detections) if detections else [],
                'memory_features': None  # Will be computed by memory network
            }

            # Compute memory features (in a real implementation, this would use the memory network)
            memory_entry['memory_features'] = np.mean([
                memory_entry['vision_features'],
                memory_entry['language_features'],
                memory_entry['lidar_features']
            ], axis=0)

            # Add to episodic memory
            self.episodic_memory.append(memory_entry)

            # Maintain memory size limit
            if len(self.episodic_memory) > self.max_memory_size:
                self.episodic_memory = self.episodic_memory[-self.max_memory_size:]

            # Update semantic memory based on experience
            self.update_semantic_memory(memory_entry)

        except Exception as e:
            self.get_logger().error(f'Error updating memory: {str(e)}')

    def update_semantic_memory(self, memory_entry):
        """Update semantic memory with learned patterns"""
        try:
            # Extract semantic patterns from experience
            command_words = memory_entry['command'].lower().split()

            for word in command_words:
                if word not in self.semantic_memory:
                    self.semantic_memory[word] = {
                        'frequency': 1,
                        'contexts': [memory_entry['vision_features']],
                        'associated_actions': [memory_entry['action']],
                        'last_used': memory_entry['timestamp']
                    }
                else:
                    self.semantic_memory[word]['frequency'] += 1
                    self.semantic_memory[word]['contexts'].append(memory_entry['vision_features'])
                    self.semantic_memory[word]['associated_actions'].append(memory_entry['action'])
                    self.semantic_memory[word]['last_used'] = memory_entry['timestamp']

                    # Limit context history
                    if len(self.semantic_memory[word]['contexts']) > 100:
                        self.semantic_memory[word]['contexts'] = self.semantic_memory[word]['contexts'][-50:]
                        self.semantic_memory[word]['associated_actions'] = self.semantic_memory[word]['associated_actions'][-50:]

        except Exception as e:
            self.get_logger().error(f'Error updating semantic memory: {str(e)}')

    def extract_vision_features(self, image_msg):
        """Extract vision features"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            with torch.no_grad():
                features = self.vision_encoder(image_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting vision features: {str(e)}')
            return None

    def extract_language_features(self, command):
        """Extract language features"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            with torch.no_grad():
                features = self.language_encoder(token_tensor)
                return features

        except Exception as e:
            self.get_logger().error(f'Error extracting language features: {str(e)}')
            return None

    def extract_lidar_features(self, scan_msg):
        """Extract LiDAR features"""
        try:
            scan_data = np.array(scan_msg.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)

            scan_tensor = torch.FloatTensor([scan_data])

            # Simple processing to create features
            features = torch.mean(scan_tensor, dim=1, keepdim=True)  # Average across all ranges
            features = features.repeat(1, 256)  # Repeat to match feature dimension

            return features

        except Exception as e:
            self.get_logger().error(f'Error extracting LiDAR features: {str(e)}')
            return torch.zeros(1, 256)  # Return zeros if error

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

    def format_detections(self, detections_msg):
        """Format detections for memory storage"""
        if not detections_msg:
            return []

        formatted_dets = []
        for detection in detections_msg.detections:
            formatted_det = {
                'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                'position': {
                    'x': detection.bbox.center.x,
                    'y': detection.bbox.center.y
                }
            }
            formatted_dets.append(formatted_det)

        return formatted_dets

def main(args=None):
    rclpy.init(args=args)
    memory_vla = MemoryAugmentedVLANode()

    try:
        rclpy.spin(memory_vla)
    except KeyboardInterrupt:
        memory_vla.get_logger().info('Memory-augmented VLA shutting down')
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

### Integration Best Practices

Implementing best practices for VLA system integration:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time

class VLABestPracticesNode(Node):
    def __init__(self):
        super().__init__('vla_best_practices')

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

        self.performance_pub = self.create_publisher(
            Float32,
            '/vla_performance',
            10
        )

        self.best_practices_pub = self.create_publisher(
            String,
            '/vla_best_practices_status',
            10
        )

        # Initialize VLA components with best practices
        self.vla_system = None
        self.initialize_best_practice_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None

        # Performance tracking
        self.processing_times = []
        self.max_processing_time = 0.1  # 100ms target
        self.frame_count = 0
        self.last_performance_report = time.time()

        # Control timer
        self.practices_timer = self.create_timer(0.05, self.best_practices_vla_loop)  # 20Hz

        self.get_logger().info('VLA best practices system initialized')

    def initialize_best_practice_components(self):
        """Initialize VLA components following best practices"""
        try:
            # Best practice: Modular architecture with clear separation of concerns
            class VLAModule(nn.Module):
                def __init__(self):
                    super(VLAModule, self).__init__()

                    # Vision module
                    self.vision_module = nn.Sequential(
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

                    # Language module
                    self.language_module = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # LiDAR module
                    self.lidar_module = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    # Fusion module with dropout for regularization
                    self.fusion_module = nn.Sequential(
                        nn.Linear(256 * 3, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),  # Best practice: Use dropout for regularization
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                    # Best practice: Initialize weights properly
                    self._initialize_weights()

                def _initialize_weights(self):
                    """Best practice: Proper weight initialization"""
                    for m in self.modules():
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)

                def forward(self, vision_input, language_input, lidar_input):
                    # Process vision
                    vision_features = self.vision_module(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)  # Best practice: Normalize features

                    # Process language
                    lang_embedded = self.language_module[0](language_input)
                    lang_lstm_out, _ = self.language_module[1](lang_embedded)
                    lang_features = self.language_module[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)  # Best practice: Normalize features

                    # Process LiDAR
                    lidar_features = self.lidar_module(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)  # Best practice: Normalize features

                    # Fuse modalities
                    combined_features = torch.cat([vision_features, lang_features, lidar_features], dim=1)
                    action_output = self.fusion_module(combined_features)

                    return action_output

            # Initialize model
            self.vla_system = VLAModule()
            self.vla_system.eval()  # Best practice: Set to eval mode for inference

            # Best practice: Use mixed precision if available (simplified for this example)
            self.use_mixed_precision = False

            self.get_logger().info('VLA system initialized with best practices')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize VLA system: {str(e)}')

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

    def best_practices_vla_loop(self):
        """Main VLA processing loop with best practices"""
        if (self.vla_system is None or
            self.current_image is None or
            self.current_command is None):
            return

        start_time = time.time()

        try:
            # Best practice: Input validation
            vision_features = self.validate_and_extract_vision_features(self.current_image)
            language_features = self.validate_and_extract_language_features(self.current_command)
            lidar_features = self.validate_and_extract_lidar_features(self.current_scan) if self.current_scan else torch.zeros(1, 256)

            if all(feat is not None for feat in [vision_features, language_features]):
                # Best practice: Use context manager for inference
                with torch.no_grad():
                    action_output = self.vla_system(vision_features, language_features, lidar_features)

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                if cmd is not None:
                    self.action_pub.publish(cmd)

                # Calculate processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Best practice: Maintain rolling average of performance
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]

                avg_processing_time = np.mean(self.processing_times)

                # Best practice: Performance monitoring and alerts
                if avg_processing_time > self.max_processing_time:
                    self.get_logger().warn(
                        f'Performance degradation detected: avg processing time {avg_processing_time:.3f}s > {self.max_processing_time:.3f}s'
                    )

                # Best practice: Publish performance metrics
                perf_msg = Float32()
                perf_msg.data = avg_processing_time
                self.performance_pub.publish(perf_msg)

                # Best practice: Regular performance reporting
                if time.time() - self.last_performance_report > 5.0:  # Every 5 seconds
                    self.publish_performance_report(avg_processing_time)
                    self.last_performance_report = time.time()

                # Best practice: Publish system status
                status_msg = String()
                status_msg.data = json.dumps({
                    'processing_time_avg': avg_processing_time,
                    'processing_time_current': processing_time,
                    'frame_count': self.frame_count,
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                })
                self.best_practices_pub.publish(status_msg)

                self.frame_count += 1

                self.get_logger().info(
                    f'VLA Best Practices - Processing Time: {processing_time:.3f}s (Avg: {avg_processing_time:.3f}s), '
                    f'Action - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in VLA best practices processing: {str(e)}')

    def validate_and_extract_vision_features(self, image_msg):
        """Validate image and extract features with best practices"""
        try:
            # Best practice: Validate input
            if image_msg.width == 0 or image_msg.height == 0:
                self.get_logger().error('Invalid image dimensions')
                return None

            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Best practice: Preprocess consistently
            image_resized = cv2.resize(cv_image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.FloatTensor(image_tensor)

            # Best practice: Ensure tensor is on correct device (CPU for this example)
            image_tensor = image_tensor  # Already on CPU

            return image_tensor

        except Exception as e:
            self.get_logger().error(f'Error validating/extracting vision features: {str(e)}')
            return None

    def validate_and_extract_language_features(self, command):
        """Validate command and extract language features with best practices"""
        try:
            # Best practice: Validate input
            if not command or len(command.strip()) == 0:
                self.get_logger().error('Empty command received')
                return None

            # Best practice: Sanitize input (basic sanitization)
            command_clean = command.strip().lower()

            tokens = command_clean.split()
            token_ids = []

            for token in tokens:
                # Best practice: Handle unknown tokens gracefully
                token_hash = hash(token) % 10000
                token_ids.append(token_hash)

            # Best practice: Pad or truncate consistently
            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            # Best practice: Ensure tensor is on correct device
            token_tensor = token_tensor  # Already on CPU

            return token_tensor

        except Exception as e:
            self.get_logger().error(f'Error validating/extracting language features: {str(e)}')
            return None

    def validate_and_extract_lidar_features(self, scan_msg):
        """Validate scan and extract LiDAR features with best practices"""
        try:
            # Best practice: Validate input
            if len(scan_msg.ranges) == 0:
                self.get_logger().error('Empty laser scan received')
                return torch.zeros(1, 256)

            scan_data = np.array(scan_msg.ranges)

            # Best practice: Handle invalid ranges consistently
            scan_data = np.nan_to_num(scan_data, nan=3.0)  # Replace NaN with max range
            scan_data = np.clip(scan_data, 0.0, 3.0)      # Clip to max range

            # Best practice: Ensure consistent size
            if len(scan_data) < 360:
                # Pad with max range values
                scan_data = np.pad(scan_data, (0, 360 - len(scan_data)), constant_values=3.0)
            elif len(scan_data) > 360:
                # Truncate to 360 points
                scan_data = scan_data[:360]

            scan_tensor = torch.FloatTensor([scan_data])

            # Best practice: Normalize features consistently
            scan_tensor = F.normalize(scan_tensor, dim=1)

            return scan_tensor

        except Exception as e:
            self.get_logger().error(f'Error validating/extracting LiDAR features: {str(e)}')
            return torch.zeros(1, 256)

    def convert_action_to_command(self, action_output):
        """Convert neural network output to robot command with safety checks"""
        if action_output is None:
            return None

        try:
            action_values = action_output[0].numpy()

            cmd = Twist()
            cmd.linear.x = float(action_values[0])
            cmd.angular.z = float(action_values[1])

            # Best practice: Safety limits
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            # Best practice: Additional safety checks
            # Check for extreme values that might indicate model confusion
            if abs(cmd.linear.x) > 0.9 and abs(cmd.angular.z) > 0.9:
                self.get_logger().warn('Extreme command values detected - possible model confusion')
                # Reduce to safer values
                cmd.linear.x *= 0.5
                cmd.angular.z *= 0.5

            return cmd

        except Exception as e:
            self.get_logger().error(f'Error converting action to command: {str(e)}')
            # Best practice: Return safe default command on error
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

    def publish_performance_report(self, avg_processing_time):
        """Publish performance report with best practices metrics"""
        performance_report = {
            'timestamp': time.time(),
            'average_processing_time': avg_processing_time,
            'target_processing_time': self.max_processing_time,
            'performance_ratio': avg_processing_time / self.max_processing_time,
            'frames_processed': self.frame_count,
            'status': 'GOOD' if avg_processing_time <= self.max_processing_time else 'WARNING'
        }

        report_msg = String()
        report_msg.data = json.dumps(performance_report, indent=2)

        self.get_logger().info(f'Performance Report:\n{json.dumps(performance_report, indent=2)}')

def main(args=None):
    rclpy.init(args=args)
    best_practices_vla = VLABestPracticesNode()

    try:
        rclpy.spin(best_practices_vla)
    except KeyboardInterrupt:
        best_practices_vla.get_logger().info('VLA best practices system shutting down')
    finally:
        # Best practice: Stop robot safely
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        best_practices_vla.action_pub.publish(cmd)

        best_practices_vla.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration with Navigation and Control Systems

Implementing VLA integration with navigation and control systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import tf2_ros
from geometry_msgs.msg import TransformStamped

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration')

        # Initialize CV bridge and TF
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/plan',
            10
        )

        self.vla_integration_pub = self.create_publisher(
            String,
            '/vla_integration_status',
            10
        )

        # Initialize integration components
        self.vla_processor = None
        self.navigation_interface = None
        self.initialize_integration_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_map = None
        self.current_goal = None
        self.current_pose = None

        # Integration state
        self.integration_mode = 'reactive'  # reactive, planned, or hybrid
        self.navigation_active = False
        self.current_plan = None

        # Control timer
        self.integration_timer = self.create_timer(0.1, self.vla_integration_loop)

        self.get_logger().info('VLA integration system initialized')

    def initialize_integration_components(self):
        """Initialize VLA integration components"""
        try:
            # VLA processing module
            class VLAProcessor(nn.Module):
                def __init__(self):
                    super(VLAProcessor, self).__init__()

                    # Vision processing
                    self.vision = nn.Sequential(
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

                    # Language processing
                    self.language = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # LiDAR processing
                    self.lidar = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    # Navigation integration module
                    self.nav_integration = nn.Sequential(
                        nn.Linear(256 * 3 + 3, 512),  # Vision + Language + LiDAR + Nav context
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 4)  # [linear_x, angular_z, goal_x, goal_y]
                    )

                def forward(self, vision_input, language_input, lidar_input, nav_context):
                    # Process modalities
                    vision_features = self.vision(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)

                    lang_embedded = self.language[0](language_input)
                    lang_lstm_out, _ = self.language[1](lang_embedded)
                    lang_features = self.language[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)

                    lidar_features = self.lidar(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)

                    # Integrate with navigation context
                    combined = torch.cat([
                        vision_features, lang_features, lidar_features, nav_context
                    ], dim=1)

                    output = self.nav_integration(combined)
                    return output

            # Navigation interface
            class NavigationInterface:
                def __init__(self, node):
                    self.node = node

                def plan_path(self, start_pose, goal_pose, map_data):
                    """Plan path from start to goal using map data"""
                    # Simplified path planning (in real implementation, use A*, RRT, etc.)
                    path = Path()
                    path.header.frame_id = 'map'

                    # Create simple straight-line path
                    steps = 10
                    for i in range(steps + 1):
                        t = i / steps
                        pose_stamped = PoseStamped()
                        pose_stamped.pose.position.x = start_pose.position.x + t * (goal_pose.position.x - start_pose.position.x)
                        pose_stamped.pose.position.y = start_pose.position.y + t * (goal_pose.position.y - start_pose.position.y)
                        pose_stamped.pose.orientation.w = 1.0  # Identity orientation

                        path.poses.append(pose_stamped)

                    return path

                def execute_navigation(self, path):
                    """Execute navigation along the path"""
                    # In a real implementation, this would interface with navigation stack
                    # For this example, we'll return a simple command
                    cmd = Twist()
                    cmd.linear.x = 0.3  # Move forward
                    cmd.angular.z = 0.0
                    return cmd

                def check_obstacles(self, lidar_data, path):
                    """Check for obstacles along the path"""
                    # Check if path is blocked by obstacles
                    if len(lidar_data) > 0:
                        min_range = min(lidar_data[np.isfinite(lidar_data)]) if any(np.isfinite(lidar_data)) else float('inf')
                        return min_range < 0.8  # Obstacle within 0.8m
                    return False

            # Initialize components
            self.vla_processor = VLAProcessor()
            self.vla_processor.eval()
            self.navigation_interface = NavigationInterface(self)

            self.get_logger().info('VLA integration components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize integration components: {str(e)}')

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
        self.current_goal = msg.pose

    def odom_callback(self, msg):
        """Process odometry"""
        self.current_pose = msg.pose.pose

    def vla_integration_loop(self):
        """Main VLA integration processing loop"""
        if (self.vla_processor is None or
            self.current_image is None or
            self.current_command is None):
            return

        try:
            # Extract features from all modalities
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.extract_language_features(self.current_command)
            lidar_features = self.extract_lidar_features(self.current_scan) if self.current_scan else torch.zeros(1, 360)

            if all(feat is not None for feat in [vision_features, language_features]):
                # Prepare navigation context
                nav_context = self.prepare_navigation_context()

                # Process through VLA system
                with torch.no_grad():
                    integration_output = self.vla_processor(
                        vision_features, language_features, lidar_features, nav_context
                    )

                # Parse output: [linear_x, angular_z, goal_x, goal_y]
                output_values = integration_output[0].numpy()

                # Extract action and potential goal
                linear_x = float(output_values[0])
                angular_z = float(output_values[1])
                goal_x = float(output_values[2])
                goal_y = float(output_values[3])

                # Determine integration mode based on command and context
                integration_mode = self.determine_integration_mode(self.current_command)

                if integration_mode == 'navigation':
                    # Use VLA output to influence navigation
                    self.execute_navigation_integration(goal_x, goal_y, linear_x, angular_z)
                elif integration_mode == 'reactive':
                    # Use VLA output directly as command
                    cmd = Twist()
                    cmd.linear.x = linear_x
                    cmd.angular.z = angular_z

                    # Apply safety limits
                    cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
                    cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

                    self.cmd_vel_pub.publish(cmd)
                else:
                    # Default reactive mode
                    cmd = Twist()
                    cmd.linear.x = linear_x * 0.5  # Scale down for safety
                    cmd.angular.z = angular_z * 0.5

                    self.cmd_vel_pub.publish(cmd)

                # Publish integration status
                integration_msg = String()
                integration_msg.data = json.dumps({
                    'integration_mode': integration_mode,
                    'command': self.current_command,
                    'vla_output': {
                        'linear_x': linear_x,
                        'angular_z': angular_z,
                        'goal_x': goal_x,
                        'goal_y': goal_y
                    },
                    'navigation_active': self.navigation_active
                })
                self.vla_integration_pub.publish(integration_msg)

                self.get_logger().info(
                    f'VLA Integration - Mode: {integration_mode}, '
                    f'Command: "{self.current_command}", '
                    f'Action - Linear: {linear_x:.2f}, Angular: {angular_z:.2f}, '
                    f'Goal: ({goal_x:.2f}, {goal_y:.2f})'
                )

        except Exception as e:
            self.get_logger().error(f'Error in VLA integration: {str(e)}')

    def prepare_navigation_context(self) -> torch.Tensor:
        """Prepare navigation context tensor"""
        try:
            # Create navigation context vector [current_x, current_y, goal_x, goal_y, has_goal]
            context_vec = np.zeros(5, dtype=np.float32)

            if self.current_pose:
                context_vec[0] = self.current_pose.position.x
                context_vec[1] = self.current_pose.position.y

            if self.current_goal:
                context_vec[2] = self.current_goal.position.x
                context_vec[3] = self.current_goal.position.y
                context_vec[4] = 1.0  # Has goal flag
            else:
                context_vec[4] = 0.0  # No goal flag

            return torch.FloatTensor([context_vec])

        except Exception as e:
            self.get_logger().error(f'Error preparing navigation context: {str(e)}')
            return torch.zeros(1, 5)

    def determine_integration_mode(self, command: str) -> str:
        """Determine integration mode based on command"""
        command_lower = command.lower()

        # Check for navigation-related commands
        navigation_keywords = [
            'go to', 'navigate to', 'move to', 'go to the', 'navigate to the',
            'move toward', 'approach', 'travel to', 'reach'
        ]

        for keyword in navigation_keywords:
            if keyword in command_lower:
                return 'navigation'

        # Check for reactive/simple movement commands
        reactive_keywords = [
            'move', 'go', 'turn', 'forward', 'backward', 'left', 'right',
            'stop', 'halt', 'pause'
        ]

        for keyword in reactive_keywords:
            if keyword in command_lower:
                return 'reactive'

        # Default to reactive
        return 'reactive'

    def execute_navigation_integration(self, goal_x: float, goal_y: float, linear_x: float, angular_z: float):
        """Execute navigation with VLA integration"""
        try:
            # Check if we have map and current pose
            if self.current_map is None or self.current_pose is None:
                # Use direct command if no navigation context
                cmd = Twist()
                cmd.linear.x = linear_x
                cmd.angular.z = angular_z
                self.cmd_vel_pub.publish(cmd)
                return

            # Create goal pose
            goal_pose = Pose()
            goal_pose.position.x = goal_x
            goal_pose.position.y = goal_y
            goal_pose.orientation.w = 1.0

            # Plan path if needed
            if self.current_plan is None or self.navigation_active:
                path = self.navigation_interface.plan_path(
                    self.current_pose, goal_pose, self.current_map
                )
                self.current_plan = path
                self.path_pub.publish(path)

            # Check for obstacles in path
            if self.current_scan:
                lidar_data = np.array(self.current_scan.ranges)
                path_blocked = self.navigation_interface.check_obstacles(lidar_data, self.current_plan)

                if path_blocked:
                    # Execute reactive avoidance behavior
                    cmd = Twist()
                    cmd.linear.x = 0.0  # Stop linear motion
                    cmd.angular.z = 0.3  # Turn to avoid
                    self.cmd_vel_pub.publish(cmd)
                else:
                    # Continue with planned navigation
                    nav_cmd = self.navigation_interface.execute_navigation(self.current_plan)
                    self.cmd_vel_pub.publish(nav_cmd)
            else:
                # No scan data, use VLA direct command
                cmd = Twist()
                cmd.linear.x = linear_x
                cmd.angular.z = angular_z
                self.cmd_vel_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f'Error in navigation integration: {str(e)}')
            # Fallback to direct command
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)

    def extract_vision_features(self, image_msg):
        """Extract vision features"""
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

    def extract_language_features(self, command):
        """Extract language features"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            max_length = 20
            if len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            else:
                token_ids = token_ids[:max_length]

            token_tensor = torch.LongTensor([token_ids])

            return token_tensor

        except Exception as e:
            self.get_logger().error(f'Error extracting language features: {str(e)}')
            return None

    def extract_lidar_features(self, scan_msg):
        """Extract LiDAR features"""
        try:
            scan_data = np.array(scan_msg.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)

            # Ensure consistent size (360 points)
            if len(scan_data) < 360:
                scan_data = np.pad(scan_data, (0, 360 - len(scan_data)), constant_values=3.0)
            elif len(scan_data) > 360:
                scan_data = scan_data[:360]

            scan_tensor = torch.FloatTensor([scan_data])

            return scan_tensor

        except Exception as e:
            self.get_logger().error(f'Error extracting LiDAR features: {str(e)}')
            return torch.zeros(1, 360)

def main(args=None):
    rclpy.init(args=args)
    vla_integration = VLAIntegrationNode()

    try:
        rclpy.spin(vla_integration)
    except KeyboardInterrupt:
        vla_integration.get_logger().info('VLA integration system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        vla_integration.cmd_vel_pub.publish(cmd)

        vla_integration.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for VLA Integration

Key best practices for VLA system integration:

1. **Modular Design**: Keep vision, language, and action components modular and loosely coupled
2. **Real-time Performance**: Optimize for real-time processing with appropriate hardware
3. **Robustness**: Handle missing or corrupted sensor data gracefully
4. **Safety**: Implement safety checks around all AI-driven decisions
5. **Memory Management**: Efficiently manage GPU and system memory
6. **Calibration**: Maintain accurate sensor calibrations for proper fusion
7. **Validation**: Continuously validate outputs against safety constraints
8. **Scalability**: Design systems that can handle increased complexity
9. **Debugging**: Provide tools for analyzing and debugging VLA decisions
10. **Evaluation**: Implement metrics for assessing VLA system performance

### Physical Grounding and Simulation-to-Real Mapping

When implementing VLA integration systems:

- **Sensor Calibration**: Ensure proper calibration between vision, language understanding, and action execution
- **Latency Management**: Account for processing delays in real-time systems
- **Hardware Constraints**: Consider computational and memory limitations on real robots
- **Environmental Variations**: Account for lighting, noise, and other environmental factors
- **Safety Systems**: Implement proper safety mechanisms around VLA decisions
- **Performance Validation**: Test performance in both simulation and real environments

### Troubleshooting VLA Integration Issues

Common VLA integration problems and solutions:

- **Performance Issues**: Profile each component separately to identify bottlenecks
- **Integration Problems**: Verify data format compatibility between components
- **Calibration Issues**: Recalibrate sensors and verify coordinate transformations
- **Memory Problems**: Monitor memory usage and implement garbage collection
- **Timing Issues**: Ensure proper synchronization between modalities
- **Safety Violations**: Implement additional safety checks and validation layers

### Summary

This chapter covered various integration patterns and best practices for Vision-Language-Action (VLA) systems in robotics. You learned about sequential, parallel, and cross-modal attention integration approaches, memory-augmented systems, and how to integrate VLA systems with navigation and control infrastructure. The chapter emphasized best practices for real-time performance, robustness, safety, and scalability. Proper VLA integration is essential for creating robots that can understand and respond to complex, multimodal commands in dynamic environments. In the next chapter, we'll explore evaluation and validation techniques for VLA systems.