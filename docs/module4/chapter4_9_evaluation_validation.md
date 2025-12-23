# Module 4: Vision–Language–Action (VLA)

## Chapter 4.9: Evaluation and Validation of VLA Systems

This chapter focuses on evaluating and validating Vision-Language-Action (VLA) systems in robotics. Proper evaluation and validation are critical for ensuring that VLA systems perform reliably, safely, and effectively in real-world robotic applications.

### Understanding VLA System Evaluation

VLA system evaluation involves multiple dimensions including:

- **Perception Accuracy**: How well the system understands visual and linguistic inputs
- **Action Quality**: How well the system executes appropriate actions
- **Integration Effectiveness**: How well vision, language, and action components work together
- **Real-time Performance**: Whether the system meets timing constraints
- **Robustness**: How well the system handles various environmental conditions
- **Safety**: Whether the system operates safely in all scenarios
- **Human Interaction**: How well the system responds to natural language commands

### VLA Evaluation Framework

The VLA evaluation framework encompasses multiple assessment levels:

```
+---------------------+
|   System-Level      |
|   (End-to-End)      |
+---------------------+
|   Integration-Level |
|   (Vision-Language- |
|   Action Fusion)    |
+---------------------+
|   Component-Level   |
|   (Individual       |
|   Modalities)       |
+---------------------+
|   Unit-Level        |
|   (Specific Tasks)  |
+---------------------+
```

### Basic VLA Evaluation Implementation

Implementing fundamental evaluation components for VLA systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import time
from datetime import datetime

class VLAEvaluationNode(Node):
    def __init__(self):
        super().__init__('vla_evaluation')

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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publishers for evaluation metrics
        self.performance_pub = self.create_publisher(
            String,
            '/vla_performance_metrics',
            10
        )

        self.accuracy_pub = self.create_publisher(
            String,
            '/vla_accuracy_metrics',
            10
        )

        self.safety_pub = self.create_publisher(
            Bool,
            '/vla_safety_status',
            10
        )

        # Initialize evaluation components
        self.vla_system = None
        self.initialize_evaluation_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_pose = None
        self.current_action = None

        # Evaluation metrics
        self.evaluation_metrics = {
            'perception_accuracy': 0.0,
            'action_success_rate': 0.0,
            'timing_performance': 0.0,
            'safety_compliance': True,
            'integration_score': 0.0
        }

        # Performance tracking
        self.processing_times = []
        self.action_history = []
        self.safety_violations = 0
        self.total_evaluations = 0

        # Evaluation timer
        self.eval_timer = self.create_timer(0.1, self.vla_evaluation_loop)

        self.get_logger().info('VLA evaluation system initialized')

    def initialize_evaluation_components(self):
        """Initialize VLA evaluation components"""
        try:
            # Simple VLA system for evaluation (simplified model)
            class VLASystem(nn.Module):
                def __init__(self):
                    super(VLASystem, self).__init__()

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

                    # Fusion and action generation
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

                def forward(self, vision_input, language_input, lidar_input):
                    # Process vision
                    vision_features = self.vision(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)

                    # Process language
                    lang_embedded = self.language[0](language_input)
                    lang_lstm_out, _ = self.language[1](lang_embedded)
                    lang_features = self.language[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)

                    # Process LiDAR
                    lidar_features = self.lidar(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)

                    # Fuse modalities
                    combined_features = torch.cat([vision_features, lang_features, lidar_features], dim=1)
                    action_output = self.fusion(combined_features)

                    return action_output

            # Initialize VLA system
            self.vla_system = VLASystem()
            self.vla_system.eval()

            # Initialize evaluation metrics
            self.evaluation_criteria = {
                'perception_threshold': 0.7,  # Minimum confidence for valid perception
                'action_threshold': 0.1,      # Maximum deviation for valid action
                'timing_threshold': 0.1,      # Maximum processing time (seconds)
                'safety_limits': {
                    'max_linear_vel': 1.0,
                    'max_angular_vel': 1.0,
                    'min_obstacle_distance': 0.5
                }
            }

            self.get_logger().info('VLA evaluation components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize evaluation components: {str(e)}')

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

    def odom_callback(self, msg):
        """Process odometry"""
        self.current_pose = msg.pose.pose

    def cmd_vel_callback(self, msg):
        """Process velocity command"""
        self.current_action = msg

    def vla_evaluation_loop(self):
        """Main VLA evaluation loop"""
        if (self.vla_system is None or
            self.current_image is None or
            self.current_command is None):
            return

        start_time = time.time()

        try:
            # Extract features from modalities
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.extract_language_features(self.current_command)
            lidar_features = self.extract_lidar_features(self.current_scan) if self.current_scan else torch.zeros(1, 360)

            if all(feat is not None for feat in [vision_features, language_features]):
                # Process through VLA system
                with torch.no_grad():
                    action_output = self.vla_system(vision_features, language_features, lidar_features)

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                # Evaluate action quality
                action_quality = self.evaluate_action_quality(cmd)
                perception_quality = self.evaluate_perception_quality(
                    vision_features, language_features, self.current_detections
                )
                timing_quality = self.evaluate_timing_performance(time.time() - start_time)
                safety_compliance = self.evaluate_safety_compliance(cmd)

                # Calculate overall metrics
                integration_score = self.calculate_integration_score(
                    perception_quality, action_quality, timing_quality
                )

                # Update evaluation metrics
                self.evaluation_metrics.update({
                    'perception_accuracy': perception_quality,
                    'action_success_rate': action_quality,
                    'timing_performance': timing_quality,
                    'safety_compliance': safety_compliance,
                    'integration_score': integration_score
                })

                # Store performance data
                self.processing_times.append(time.time() - start_time)
                if len(self.processing_times) > 100:  # Keep last 100 measurements
                    self.processing_times = self.processing_times[-100:]

                # Publish evaluation results
                self.publish_evaluation_metrics()

                # Log evaluation results
                self.get_logger().info(
                    f'VLA Evaluation - Perception: {perception_quality:.3f}, '
                    f'Action: {action_quality:.3f}, '
                    f'Timing: {timing_quality:.3f}, '
                    f'Safety: {safety_compliance}, '
                    f'Integration: {integration_score:.3f}, '
                    f'Processing Time: {(time.time() - start_time)*1000:.1f}ms'
                )

                self.total_evaluations += 1

        except Exception as e:
            self.get_logger().error(f'Error in VLA evaluation: {str(e)}')

    def extract_vision_features(self, image_msg):
        """Extract features from image"""
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
        """Extract features from language command"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate to fixed length
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
        """Extract features from LiDAR scan"""
        try:
            scan_data = np.array(scan_msg.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)

            # Ensure consistent size
            if len(scan_data) < 360:
                scan_data = np.pad(scan_data, (0, 360 - len(scan_data)), constant_values=3.0)
            elif len(scan_data) > 360:
                scan_data = scan_data[:360]

            scan_tensor = torch.FloatTensor([scan_data])

            return scan_tensor

        except Exception as e:
            self.get_logger().error(f'Error extracting LiDAR features: {str(e)}')
            return torch.zeros(1, 360)

    def evaluate_perception_quality(self, vision_features, language_features, detections):
        """Evaluate perception quality"""
        try:
            # In a real system, this would evaluate object detection accuracy, language understanding, etc.
            # For this example, we'll use a simplified approach

            # Evaluate based on presence of detections
            detection_confidence = 0.0
            if detections and detections.detections:
                avg_confidence = np.mean([
                    det.results[0].hypothesis.score if det.results else 0.0
                    for det in detections.detections
                ])
                detection_confidence = avg_confidence

            # Evaluate language understanding (simplified)
            language_confidence = min(1.0, len(self.current_command.split()) / 10.0)

            # Combine scores
            perception_score = (detection_confidence + language_confidence) / 2.0

            return perception_score

        except Exception as e:
            self.get_logger().error(f'Error evaluating perception quality: {str(e)}')
            return 0.0

    def evaluate_action_quality(self, cmd):
        """Evaluate action quality"""
        try:
            if cmd is None:
                return 0.0

            # Check if action is reasonable (not too extreme)
            linear_magnitude = abs(cmd.linear.x)
            angular_magnitude = abs(cmd.angular.z)

            # Normalize to 0-1 range (lower is better for extremes)
            linear_score = max(0.0, 1.0 - linear_magnitude)
            angular_score = max(0.0, 1.0 - angular_magnitude)

            # Average the scores
            action_score = (linear_score + angular_score) / 2.0

            return action_score

        except Exception as e:
            self.get_logger().error(f'Error evaluating action quality: {str(e)}')
            return 0.0

    def evaluate_timing_performance(self, processing_time):
        """Evaluate timing performance"""
        try:
            # Calculate score based on processing time threshold
            if processing_time <= self.evaluation_criteria['timing_threshold']:
                timing_score = 1.0
            else:
                # Score decreases as time increases beyond threshold
                excess_time = processing_time - self.evaluation_criteria['timing_threshold']
                timing_score = max(0.0, 1.0 - (excess_time / self.evaluation_criteria['timing_threshold']))

            return timing_score

        except Exception as e:
            self.get_logger().error(f'Error evaluating timing performance: {str(e)}')
            return 0.0

    def evaluate_safety_compliance(self, cmd):
        """Evaluate safety compliance"""
        try:
            if cmd is None:
                return False

            # Check velocity limits
            if (abs(cmd.linear.x) > self.evaluation_criteria['safety_limits']['max_linear_vel'] or
                abs(cmd.angular.z) > self.evaluation_criteria['safety_limits']['max_angular_vel']):
                return False

            # Check for obstacles if scan data available
            if self.current_scan:
                ranges = np.array(self.current_scan.ranges)
                finite_ranges = ranges[np.isfinite(ranges)]
                if len(finite_ranges) > 0:
                    min_range = np.min(finite_ranges)
                    if min_range < self.evaluation_criteria['safety_limits']['min_obstacle_distance']:
                        # Robot is commanded to move forward toward close obstacle
                        if cmd.linear.x > 0.1:  # Moving forward
                            return False

            return True

        except Exception as e:
            self.get_logger().error(f'Error evaluating safety compliance: {str(e)}')
            return False

    def calculate_integration_score(self, perception_score, action_score, timing_score):
        """Calculate overall integration score"""
        try:
            # Weighted combination of different scores
            weights = {
                'perception': 0.4,
                'action': 0.3,
                'timing': 0.2,
                'safety': 0.1
            }

            integration_score = (
                weights['perception'] * perception_score +
                weights['action'] * action_score +
                weights['timing'] * timing_score +
                weights['safety'] * float(self.evaluate_safety_compliance(self.current_action))
            )

            return integration_score

        except Exception as e:
            self.get_logger().error(f'Error calculating integration score: {str(e)}')
            return 0.0

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

    def publish_evaluation_metrics(self):
        """Publish evaluation metrics"""
        try:
            # Performance metrics
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0

            metrics_msg = String()
            metrics_msg.data = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'evaluation_metrics': self.evaluation_metrics,
                'performance': {
                    'avg_processing_time': avg_processing_time,
                    'processing_time_samples': len(self.processing_times),
                    'min_processing_time': min(self.processing_times) if self.processing_times else 0.0,
                    'max_processing_time': max(self.processing_times) if self.processing_times else 0.0
                },
                'safety': {
                    'violations_count': self.safety_violations,
                    'total_evaluations': self.total_evaluations
                },
                'integration_score': self.evaluation_metrics['integration_score']
            })

            self.performance_pub.publish(metrics_msg)

            # Accuracy metrics
            accuracy_msg = String()
            accuracy_msg.data = json.dumps({
                'perception_accuracy': self.evaluation_metrics['perception_accuracy'],
                'action_success_rate': self.evaluation_metrics['action_success_rate'],
                'timing_performance': self.evaluation_metrics['timing_performance']
            })
            self.accuracy_pub.publish(accuracy_msg)

            # Safety status
            safety_msg = Bool()
            safety_msg.data = self.evaluation_metrics['safety_compliance']
            self.safety_pub.publish(safety_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing evaluation metrics: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    vla_eval = VLAEvaluationNode()

    try:
        rclpy.spin(vla_eval)
    except KeyboardInterrupt:
        vla_eval.get_logger().info('VLA evaluation system shutting down')
    finally:
        vla_eval.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced VLA Validation Techniques

Implementing more sophisticated validation techniques for VLA systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import time
from collections import deque
import statistics

class AdvancedVLANode(Node):
    def __init__(self):
        super().__init__('advanced_vla_validation')

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

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # Publishers for advanced validation metrics
        self.validation_pub = self.create_publisher(
            String,
            '/advanced_vla_validation',
            10
        )

        self.confidence_pub = self.create_publisher(
            Float32MultiArray,
            '/vla_confidence_scores',
            10
        )

        self.robustness_pub = self.create_publisher(
            String,
            '/vla_robustness_metrics',
            10
        )

        # Initialize advanced validation components
        self.vla_model = None
        self.confidence_estimator = None
        self.uncertainty_quantifier = None
        self.initialize_advanced_validation_components()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_goal = None
        self.current_path = None

        # Advanced validation metrics
        self.validation_metrics = {
            'confidence_scores': [],
            'uncertainty_metrics': [],
            'robustness_score': 0.0,
            'consistency_score': 0.0,
            'reliability_index': 0.0
        }

        # Validation history
        self.validation_history = deque(maxlen=100)
        self.action_consistency_buffer = deque(maxlen=50)
        self.confidence_history = deque(maxlen=100)

        # Validation parameters
        self.confidence_threshold = 0.7
        self.uncertainty_threshold = 0.3
        self.robustness_window = 10

        # Control timer
        self.advanced_timer = self.create_timer(0.1, self.advanced_vla_validation_loop)

        self.get_logger().info('Advanced VLA validation system initialized')

    def initialize_advanced_validation_components(self):
        """Initialize advanced validation components"""
        try:
            # VLA model with uncertainty estimation
            class UncertaintyAwareVLA(nn.Module):
                def __init__(self):
                    super(UncertaintyAwareVLA, self).__init__()

                    # Vision encoder with uncertainty estimation
                    self.vision_encoder = nn.Sequential(
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

                    # Uncertainty estimation for vision
                    self.vision_uncertainty = nn.Sequential(
                        nn.Linear(256 * 8 * 8, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),  # Uncertainty score
                        nn.Sigmoid()  # Normalize to [0,1]
                    )

                    # Language encoder with uncertainty
                    self.language_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    self.language_uncertainty = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )

                    # LiDAR encoder with uncertainty
                    self.lidar_encoder = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    self.lidar_uncertainty = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )

                    # Fusion with uncertainty
                    self.fusion = nn.Sequential(
                        nn.Linear(256 * 3, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                    # Overall uncertainty estimator
                    self.overall_uncertainty = nn.Sequential(
                        nn.Linear(256 * 3, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                    )

                def forward(self, vision_input, language_input, lidar_input):
                    # Process vision with uncertainty
                    vision_features = self.vision_encoder(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)
                    vision_uncertainty = self.vision_uncertainty(vision_features)

                    # Process language with uncertainty
                    lang_embedded = self.language_encoder[0](language_input)
                    lang_lstm_out, _ = self.language_encoder[1](lang_embedded)
                    lang_features = self.language_encoder[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)
                    language_uncertainty = self.language_uncertainty(lang_features)

                    # Process LiDAR with uncertainty
                    lidar_features = self.lidar_encoder(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)
                    lidar_uncertainty = self.lidar_uncertainty(lidar_features)

                    # Fuse modalities
                    combined_features = torch.cat([vision_features, lang_features, lidar_features], dim=1)
                    action_output = self.fusion(combined_features)

                    # Calculate overall uncertainty
                    overall_uncertainty = self.overall_uncertainty(combined_features)

                    return action_output, {
                        'vision_uncertainty': vision_uncertainty,
                        'language_uncertainty': language_uncertainty,
                        'lidar_uncertainty': lidar_uncertainty,
                        'overall_uncertainty': overall_uncertainty
                    }

            # Initialize models
            self.vla_model = UncertaintyAwareVLA()
            self.vla_model.eval()

            self.get_logger().info('Advanced VLA validation components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced validation components: {str(e)}')

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

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg

    def path_callback(self, msg):
        """Process navigation path"""
        self.current_path = msg

    def advanced_vla_validation_loop(self):
        """Main advanced VLA validation loop"""
        if (self.vla_model is None or
            self.current_image is None or
            self.current_command is None):
            return

        start_time = time.time()

        try:
            # Extract features with uncertainty estimation
            vision_features = self.extract_vision_features(self.current_image)
            language_features = self.extract_language_features(self.current_command)
            lidar_features = self.extract_lidar_features(self.current_scan) if self.current_scan else torch.zeros(1, 360)

            if all(feat is not None for feat in [vision_features, language_features]):
                # Process with uncertainty estimation
                with torch.no_grad():
                    action_output, uncertainty_info = self.vla_model(
                        vision_features, language_features, lidar_features
                    )

                # Convert to robot command
                cmd = self.convert_action_to_command(action_output)

                # Validate with advanced techniques
                confidence_score = self.calculate_confidence_score(uncertainty_info)
                robustness_score = self.assess_robustness(cmd)
                consistency_score = self.assess_consistency(cmd)
                reliability_index = self.calculate_reliability_index(uncertainty_info)

                # Update validation metrics
                self.validation_metrics.update({
                    'confidence_scores': [confidence_score],
                    'uncertainty_metrics': [uncertainty_info],
                    'robustness_score': robustness_score,
                    'consistency_score': consistency_score,
                    'reliability_index': reliability_index
                })

                # Store validation results
                validation_result = {
                    'timestamp': time.time(),
                    'confidence': confidence_score,
                    'robustness': robustness_score,
                    'consistency': consistency_score,
                    'reliability': reliability_index,
                    'uncertainties': {
                        'vision': float(uncertainty_info['vision_uncertainty'].mean()),
                        'language': float(uncertainty_info['language_uncertainty'].mean()),
                        'lidar': float(uncertainty_info['lidar_uncertainty'].mean()),
                        'overall': float(uncertainty_info['overall_uncertainty'].mean())
                    },
                    'action': {
                        'linear_x': float(cmd.linear.x) if cmd else 0.0,
                        'angular_z': float(cmd.angular.z) if cmd else 0.0
                    }
                }

                self.validation_history.append(validation_result)
                self.confidence_history.append(confidence_score)

                # Publish validation results
                self.publish_advanced_validation(validation_result)

                # Log validation results
                self.get_logger().info(
                    f'Advanced VLA Validation - Confidence: {confidence_score:.3f}, '
                    f'Robustness: {robustness_score:.3f}, '
                    f'Consistency: {consistency_score:.3f}, '
                    f'Reliability: {reliability_index:.3f}, '
                    f'Processing Time: {(time.time() - start_time)*1000:.1f}ms'
                )

        except Exception as e:
            self.get_logger().error(f'Error in advanced VLA validation: {str(e)}')

    def calculate_confidence_score(self, uncertainty_info: Dict) -> float:
        """Calculate confidence score based on uncertainty information"""
        try:
            # Lower uncertainty means higher confidence
            vision_uncertainty = float(uncertainty_info['vision_uncertainty'].mean())
            language_uncertainty = float(uncertainty_info['language_uncertainty'].mean())
            lidar_uncertainty = float(uncertainty_info['lidar_uncertainty'].mean())
            overall_uncertainty = float(uncertainty_info['overall_uncertainty'].mean())

            # Convert uncertainties to confidence (1 - uncertainty)
            vision_confidence = 1.0 - vision_uncertainty
            language_confidence = 1.0 - language_uncertainty
            lidar_confidence = 1.0 - lidar_uncertainty
            overall_confidence = 1.0 - overall_uncertainty

            # Weighted average confidence
            weights = {
                'vision': 0.4,
                'language': 0.3,
                'lidar': 0.2,
                'overall': 0.1
            }

            confidence_score = (
                weights['vision'] * vision_confidence +
                weights['language'] * language_confidence +
                weights['lidar'] * lidar_confidence +
                weights['overall'] * overall_confidence
            )

            return max(0.0, min(1.0, confidence_score))  # Clamp to [0,1]

        except Exception as e:
            self.get_logger().error(f'Error calculating confidence score: {str(e)}')
            return 0.0

    def assess_robustness(self, cmd: Twist) -> float:
        """Assess the robustness of the action command"""
        try:
            if cmd is None:
                return 0.0

            # Robustness assessment based on action characteristics
            linear_magnitude = abs(cmd.linear.x)
            angular_magnitude = abs(cmd.angular.z)

            # Actions with moderate velocities are more robust than extreme ones
            linear_robustness = 1.0 - min(1.0, linear_magnitude)  # Lower velocity = higher robustness
            angular_robustness = 1.0 - min(1.0, angular_magnitude)

            # Combined robustness score
            robustness_score = (linear_robustness + angular_robustness) / 2.0

            return robustness_score

        except Exception as e:
            self.get_logger().error(f'Error assessing robustness: {str(e)}')
            return 0.0

    def assess_consistency(self, cmd: Twist) -> float:
        """Assess consistency of action with previous actions"""
        try:
            if cmd is None or len(self.action_consistency_buffer) == 0:
                return 1.0  # No comparison possible, assume consistency

            # Calculate consistency with recent actions
            recent_actions = list(self.action_consistency_buffer)
            current_action = np.array([cmd.linear.x, cmd.angular.z])

            # Calculate average of recent actions
            if recent_actions:
                avg_action = np.mean(recent_actions, axis=0)
                # Calculate consistency as inverse of difference
                action_diff = np.linalg.norm(current_action - avg_action)
                consistency_score = max(0.0, 1.0 - action_diff)  # Higher difference = lower consistency
            else:
                consistency_score = 1.0

            # Store current action for future consistency checks
            self.action_consistency_buffer.append(current_action.tolist())

            return consistency_score

        except Exception as e:
            self.get_logger().error(f'Error assessing consistency: {str(e)}')
            return 0.0

    def calculate_reliability_index(self, uncertainty_info: Dict) -> float:
        """Calculate reliability index based on uncertainty consistency"""
        try:
            # Calculate variance of recent uncertainty estimates
            if len(self.confidence_history) > 1:
                recent_confidence_variance = statistics.variance(self.confidence_history)
                # Lower variance indicates more reliable uncertainty estimates
                reliability_score = max(0.0, 1.0 - recent_confidence_variance)
            else:
                reliability_score = 1.0  # Not enough data to assess reliability

            return reliability_score

        except Exception as e:
            self.get_logger().error(f'Error calculating reliability index: {str(e)}')
            return 0.0

    def publish_advanced_validation(self, validation_result: Dict):
        """Publish advanced validation results"""
        try:
            # Publish detailed validation report
            validation_msg = String()
            validation_msg.data = json.dumps(validation_result)
            self.validation_pub.publish(validation_msg)

            # Publish confidence scores
            confidence_msg = Float32MultiArray()
            confidence_msg.data = [
                validation_result['confidence'],
                validation_result['robustness'],
                validation_result['consistency'],
                validation_result['reliability']
            ]
            self.confidence_pub.publish(confidence_msg)

            # Publish robustness metrics
            robustness_msg = String()
            robustness_msg.data = json.dumps({
                'timestamp': validation_result['timestamp'],
                'uncertainty_breakdown': validation_result['uncertainties'],
                'action_stability': self.calculate_action_stability(),
                'environment_variability': self.assess_environment_variability()
            })
            self.robustness_pub.publish(robustness_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing advanced validation: {str(e)}')

    def calculate_action_stability(self) -> float:
        """Calculate action stability over recent history"""
        try:
            if len(self.action_consistency_buffer) < 2:
                return 1.0

            actions = np.array(self.action_consistency_buffer)

            # Calculate variance of actions (lower variance = more stable)
            action_variance = np.var(actions, axis=0)
            stability_score = max(0.0, 1.0 - np.mean(action_variance))

            return stability_score

        except Exception as e:
            self.get_logger().error(f'Error calculating action stability: {str(e)}')
            return 0.0

    def assess_environment_variability(self) -> float:
        """Assess environmental variability based on sensor data"""
        try:
            # This would normally compare current sensor data with historical data
            # For this example, we'll return a placeholder value
            # In a real implementation, this would analyze sensor data variance

            if self.current_scan:
                # Analyze scan data for environmental changes
                ranges = np.array(self.current_scan.ranges)
                valid_ranges = ranges[np.isfinite(ranges)]

                if len(valid_ranges) > 0:
                    # Calculate variance in range readings (higher variance = more dynamic environment)
                    range_variance = np.var(valid_ranges)
                    environment_stability = max(0.0, 1.0 - range_variance / 10.0)  # Normalize
                    return environment_stability

            return 0.8  # Default medium stability

        except Exception as e:
            self.get_logger().error(f'Error assessing environment variability: {str(e)}')
            return 0.5

def main(args=None):
    rclpy.init(args=args)
    advanced_eval = AdvancedVLAEvaluationNode()

    try:
        rclpy.spin(advanced_eval)
    except KeyboardInterrupt:
        advanced_eval.get_logger().info('Advanced VLA validation shutting down')
    finally:
        advanced_eval.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Simulation-to-Real Validation Framework

Implementing validation for simulation-to-real transfer:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import time

class SimulationToRealValidationNode(Node):
    def __init__(self):
        super().__init__('simulation_to_real_validation')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers for both simulation and real data
        self.sim_image_sub = self.create_subscription(
            Image,
            '/sim/camera/image_raw',
            self.sim_image_callback,
            10
        )

        self.real_image_sub = self.create_subscription(
            Image,
            '/real/camera/image_raw',
            self.real_image_callback,
            10
        )

        self.sim_scan_sub = self.create_subscription(
            LaserScan,
            '/sim/scan',
            self.sim_scan_callback,
            10
        )

        self.real_scan_sub = self.create_subscription(
            LaserScan,
            '/real/scan',
            self.real_scan_callback,
            10
        )

        self.sim_detection_sub = self.create_subscription(
            Detection2DArray,
            '/sim/detections',
            self.sim_detection_callback,
            10
        )

        self.real_detection_sub = self.create_subscription(
            Detection2DArray,
            '/real/detections',
            self.real_detection_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        # Publishers for validation metrics
        self.transferability_pub = self.create_publisher(
            String,
            '/sim2real_transferability',
            10
        )

        self.domain_gap_pub = self.create_publisher(
            String,
            '/domain_gap_metrics',
            10
        )

        self.adaptation_needed_pub = self.create_publisher(
            Bool,
            '/adaptation_needed',
            10
        )

        # Initialize validation components
        self.sim_vla_model = None
        self.real_vla_model = None
        self.domain_adaptation_module = None
        self.initialize_sim2real_components()

        # State variables for both sim and real
        self.sim_image = None
        self.real_image = None
        self.sim_scan = None
        self.real_scan = None
        self.sim_detections = None
        self.real_detections = None
        self.current_command = None

        # Validation metrics
        self.sim2real_metrics = {
            'domain_gap': 0.0,
            'transferability_score': 0.0,
            'adaptation_required': False,
            'performance_delta': 0.0
        }

        # Feature buffers for domain gap analysis
        self.sim_features_buffer = []
        self.real_features_buffer = []
        self.max_buffer_size = 100

        # Control timer
        self.sim2real_timer = self.create_timer(0.2, self.sim2real_validation_loop)  # Slower for feature analysis

        self.get_logger().info('Simulation-to-real validation system initialized')

    def initialize_sim2real_components(self):
        """Initialize simulation-to-real validation components"""
        try:
            # Simulated VLA model
            class SimulatedVLAModel(nn.Module):
                def __init__(self):
                    super(SimulatedVLAModel, self).__init__()

                    # Vision encoder for simulation
                    self.vision_encoder = nn.Sequential(
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

                    # Language encoder
                    self.language_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # LiDAR encoder for simulation (cleaner data)
                    self.sim_lidar_encoder = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    # Action decoder
                    self.action_decoder = nn.Sequential(
                        nn.Linear(256 * 3, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)  # [linear_x, angular_z]
                    )

                def forward(self, vision_input, language_input, lidar_input):
                    vision_features = self.vision_encoder(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)

                    lang_embedded = self.language_encoder[0](language_input)
                    lang_lstm_out, _ = self.language_encoder[1](lang_embedded)
                    lang_features = self.language_encoder[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)

                    lidar_features = self.sim_lidar_encoder(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)

                    combined_features = torch.cat([vision_features, lang_features, lidar_features], dim=1)
                    action_output = self.action_decoder(combined_features)

                    return action_output, combined_features

            # Real-world VLA model (with noise adaptation)
            class RealWorldVLAModel(nn.Module):
                def __init__(self):
                    super(RealWorldVLAModel, self).__init__()

                    # Vision encoder for real world (handles noise better)
                    self.vision_encoder = nn.Sequential(
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

                    # Language encoder
                    self.language_encoder = nn.Sequential(
                        nn.Embedding(10000, 256),
                        nn.LSTM(256, 256, batch_first=True),
                        nn.Linear(256, 256)
                    )

                    # LiDAR encoder for real world (handles noise)
                    self.real_lidar_encoder = nn.Sequential(
                        nn.Linear(360, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256)
                    )

                    # Action decoder
                    self.action_decoder = nn.Sequential(
                        nn.Linear(256 * 3, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),  # Higher dropout for real-world robustness
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2)
                    )

                def forward(self, vision_input, language_input, lidar_input):
                    vision_features = self.vision_encoder(vision_input)
                    vision_features = vision_features.view(vision_features.size(0), -1)
                    vision_features = F.normalize(vision_features, dim=1)

                    lang_embedded = self.language_encoder[0](language_input)
                    lang_lstm_out, _ = self.language_encoder[1](lang_embedded)
                    lang_features = self.language_encoder[2](lang_lstm_out[:, -1, :])
                    lang_features = F.normalize(lang_features, dim=1)

                    lidar_features = self.real_lidar_encoder(lidar_input)
                    lidar_features = F.normalize(lidar_features, dim=1)

                    combined_features = torch.cat([vision_features, lang_features, lidar_features], dim=1)
                    action_output = self.action_decoder(combined_features)

                    return action_output, combined_features

            # Domain adaptation module
            class DomainAdaptationModule(nn.Module):
                def __init__(self):
                    super(DomainAdaptationModule, self).__init__()

                    # Feature alignment network
                    self.feature_aligner = nn.Sequential(
                        nn.Linear(256 * 3, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256 * 3)  # Same dimension as features
                    )

                def forward(self, sim_features):
                    # Align simulation features to real-world distribution
                    aligned_features = self.feature_aligner(sim_features)
                    return aligned_features

            # Initialize models
            self.sim_vla_model = SimulatedVLAModel()
            self.real_vla_model = RealWorldVLAModel()
            self.domain_adaptation_module = DomainAdaptationModule()

            # Set to evaluation mode
            self.sim_vla_model.eval()
            self.real_vla_model.eval()
            self.domain_adaptation_module.eval()

            self.get_logger().info('Sim-to-real validation components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize sim-to-real components: {str(e)}')

    def sim_image_callback(self, msg):
        """Process simulated image"""
        self.sim_image = msg

    def real_image_callback(self, msg):
        """Process real image"""
        self.real_image = msg

    def sim_scan_callback(self, msg):
        """Process simulated scan"""
        self.sim_scan = msg

    def real_scan_callback(self, msg):
        """Process real scan"""
        self.real_scan = msg

    def sim_detection_callback(self, msg):
        """Process simulated detections"""
        self.sim_detections = msg

    def real_detection_callback(self, msg):
        """Process real detections"""
        self.real_detections = msg

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data

    def sim2real_validation_loop(self):
        """Main simulation-to-real validation loop"""
        if (self.sim_vla_model is None or
            self.current_command is None or
            self.sim_image is None or
            self.real_image is None):
            return

        try:
            # Extract features from both simulation and real environments
            sim_vision_features = self.extract_vision_features(self.sim_image)
            real_vision_features = self.extract_vision_features(self.real_image)
            language_features = self.extract_language_features(self.current_command)

            sim_lidar_features = self.extract_lidar_features(self.sim_scan) if self.sim_scan else torch.zeros(1, 360)
            real_lidar_features = self.extract_lidar_features(self.real_scan) if self.real_scan else torch.zeros(1, 360)

            if all(feat is not None for feat in [sim_vision_features, real_vision_features, language_features]):
                # Get actions from both models
                with torch.no_grad():
                    sim_action, sim_features = self.sim_vla_model(
                        sim_vision_features, language_features, sim_lidar_features
                    )
                    real_action, real_features = self.real_vla_model(
                        real_vision_features, language_features, real_lidar_features
                    )

                # Calculate domain gap
                domain_gap = self.calculate_domain_gap(sim_features, real_features)

                # Assess transferability
                transferability_score = self.assess_transferability(sim_action, real_action)

                # Check if adaptation is needed
                adaptation_needed = self.is_adaptation_needed(domain_gap, transferability_score)

                # Calculate performance delta
                performance_delta = self.calculate_performance_delta(sim_action, real_action)

                # Update metrics
                self.sim2real_metrics.update({
                    'domain_gap': domain_gap,
                    'transferability_score': transferability_score,
                    'adaptation_required': adaptation_needed,
                    'performance_delta': performance_delta
                })

                # Store features for ongoing domain gap analysis
                self.sim_features_buffer.append(sim_features.cpu().numpy())
                self.real_features_buffer.append(real_features.cpu().numpy())

                if len(self.sim_features_buffer) > self.max_buffer_size:
                    self.sim_features_buffer.pop(0)
                if len(self.real_features_buffer) > self.max_buffer_size:
                    self.real_features_buffer.pop(0)

                # Publish validation results
                self.publish_sim2real_validation()

                self.get_logger().info(
                    f'Sim2Real Validation - Domain Gap: {domain_gap:.3f}, '
                    f'Transferability: {transferability_score:.3f}, '
                    f'Adaptation Needed: {adaptation_needed}, '
                    f'Performance Delta: {performance_delta:.3f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in sim-to-real validation: {str(e)}')

    def calculate_domain_gap(self, sim_features: torch.Tensor, real_features: torch.Tensor) -> float:
        """Calculate domain gap between simulation and real features"""
        try:
            # Calculate distance between feature distributions
            # Using simple Euclidean distance between mean features as an approximation
            sim_mean = torch.mean(sim_features, dim=0)
            real_mean = torch.mean(real_features, dim=0)

            # Calculate Euclidean distance
            distance = torch.norm(sim_mean - real_mean, p=2)

            # Normalize to [0, 1] range (arbitrary normalization factor)
            domain_gap = float(torch.tanh(distance / 10.0))  # tanh to bound between 0 and 1

            return domain_gap

        except Exception as e:
            self.get_logger().error(f'Error calculating domain gap: {str(e)}')
            return 1.0  # High gap on error

    def assess_transferability(self, sim_action: torch.Tensor, real_action: torch.Tensor) -> float:
        """Assess how well simulation action transfers to real world"""
        try:
            # Calculate similarity between simulation and real actions
            action_diff = torch.norm(sim_action - real_action, p=2)

            # Convert to transferability score (inverse relationship)
            transferability = float(torch.exp(-action_diff))  # Exponential decay

            return transferability

        except Exception as e:
            self.get_logger().error(f'Error assessing transferability: {str(e)}')
            return 0.0

    def is_adaptation_needed(self, domain_gap: float, transferability_score: float) -> bool:
        """Determine if domain adaptation is needed"""
        # Define thresholds
        domain_gap_threshold = 0.3  # If domain gap is above this, adaptation may be needed
        transferability_threshold = 0.7  # If transferability is below this, adaptation may be needed

        # Adaptation is needed if domain gap is high OR transferability is low
        adaptation_needed = (domain_gap > domain_gap_threshold or
                           transferability_score < transferability_threshold)

        return adaptation_needed

    def calculate_performance_delta(self, sim_action: torch.Tensor, real_action: torch.Tensor) -> float:
        """Calculate performance delta between sim and real"""
        try:
            # Calculate difference in action magnitudes
            sim_magnitude = torch.norm(sim_action, p=2)
            real_magnitude = torch.norm(real_action, p=2)

            # Calculate relative difference
            if sim_magnitude > 0:
                delta = abs(float(real_magnitude - sim_magnitude)) / float(sim_magnitude)
            else:
                delta = float(real_magnitude)  # If sim action is zero, use real magnitude

            return delta

        except Exception as e:
            self.get_logger().error(f'Error calculating performance delta: {str(e)}')
            return 1.0

    def publish_sim2real_validation(self):
        """Publish simulation-to-real validation results"""
        try:
            # Transferability report
            transferability_msg = String()
            transferability_msg.data = json.dumps({
                'timestamp': time.time(),
                'domain_gap': self.sim2real_metrics['domain_gap'],
                'transferability_score': self.sim2real_metrics['transferability_score'],
                'performance_delta': self.sim2real_metrics['performance_delta'],
                'adaptation_required': self.sim2real_metrics['adaptation_required'],
                'feature_buffer_sizes': {
                    'sim': len(self.sim_features_buffer),
                    'real': len(self.real_features_buffer)
                }
            })
            self.transferability_pub.publish(transferability_msg)

            # Domain gap metrics
            domain_gap_msg = String()
            domain_gap_msg.data = json.dumps({
                'domain_gap_measure': self.sim2real_metrics['domain_gap'],
                'feature_alignment_status': 'needs_alignment' if self.sim2real_metrics['adaptation_required'] else 'aligned',
                'sim_features_shape': self.sim_features_buffer[-1].shape if self.sim_features_buffer else None,
                'real_features_shape': self.real_features_buffer[-1].shape if self.real_features_buffer else None
            })
            self.domain_gap_pub.publish(domain_gap_msg)

            # Adaptation needed flag
            adaptation_msg = Bool()
            adaptation_msg.data = self.sim2real_metrics['adaptation_required']
            self.adaptation_needed_pub.publish(adaptation_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing sim2real validation: {str(e)}')

    def extract_vision_features(self, image_msg):
        """Extract vision features from image message"""
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
        """Extract language features from command"""
        try:
            tokens = command.lower().split()
            token_ids = [hash(token) % 10000 for token in tokens]

            # Pad/truncate
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
        """Extract LiDAR features from scan message"""
        try:
            scan_data = np.array(scan_msg.ranges)
            scan_data = np.nan_to_num(scan_data, nan=3.0)
            scan_data = np.clip(scan_data, 0.0, 3.0)

            # Ensure consistent size
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
    sim2real_eval = SimulationToRealValidationNode()

    try:
        rclpy.spin(sim2real_eval)
    except KeyboardInterrupt:
        sim2real_eval.get_logger().info('Sim-to-real validation shutting down')
    finally:
        sim2real_eval.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Validation Metrics and Reporting

Creating comprehensive validation metrics and reporting:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
import numpy as np
import json
import time
from datetime import datetime
from collections import deque
import statistics

class VLAValidationMetricsNode(Node):
    def __init__(self):
        super().__init__('vla_validation_metrics')

        # Publishers for different metric types
        self.performance_pub = self.create_publisher(
            String,
            '/vla_performance_report',
            10
        )

        self.accuracy_pub = self.create_publisher(
            String,
            '/vla_accuracy_report',
            10
        )

        self.safety_pub = self.create_publisher(
            String,
            '/vla_safety_report',
            10
        )

        self.comprehensive_pub = self.create_publisher(
            String,
            '/vla_comprehensive_report',
            10
        )

        # Initialize validation metrics tracking
        self.initialize_validation_tracking()

        # Validation timer
        self.metrics_timer = self.create_timer(1.0, self.generate_validation_report)

        self.get_logger().info('VLA validation metrics system initialized')

    def initialize_validation_tracking(self):
        """Initialize validation metrics tracking"""
        # Performance metrics
        self.timing_buffer = deque(maxlen=100)
        self.throughput_buffer = deque(maxlen=100)

        # Accuracy metrics
        self.action_success_buffer = deque(maxlen=100)
        self.perception_accuracy_buffer = deque(maxlen=100)

        # Safety metrics
        self.safety_violation_buffer = deque(maxlen=100)
        self.emergency_stops_buffer = deque(maxlen=100)

        # Robustness metrics
        self.error_recovery_buffer = deque(maxlen=100)
        self.system_uptime_buffer = deque(maxlen=100)

        # Initialize counters
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.safety_violations = 0
        self.emergency_stops = 0
        self.error_recoveries = 0

        # Timestamps
        self.start_time = time.time()
        self.last_report_time = time.time()

    def add_timing_measurement(self, processing_time: float):
        """Add timing measurement to buffer"""
        self.timing_buffer.append(processing_time)

    def add_action_success(self, success: bool):
        """Add action success measurement"""
        self.action_success_buffer.append(1 if success else 0)

    def add_perception_accuracy(self, accuracy: float):
        """Add perception accuracy measurement"""
        self.perception_accuracy_buffer.append(accuracy)

    def add_safety_violation(self):
        """Record safety violation"""
        self.safety_violations += 1
        self.safety_violation_buffer.append(time.time())

    def add_emergency_stop(self):
        """Record emergency stop"""
        self.emergency_stops += 1
        self.emergency_stops_buffer.append(time.time())

    def add_error_recovery(self):
        """Record error recovery"""
        self.error_recoveries += 1
        self.error_recovery_buffer.append(time.time())

    def add_evaluation_result(self, success: bool):
        """Add evaluation result"""
        self.total_evaluations += 1
        if success:
            self.successful_evaluations += 1
        else:
            self.failed_evaluations += 1

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        current_time = time.time()
        elapsed_time = current_time - self.last_report_time
        self.last_report_time = current_time

        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics()

        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics()

        # Calculate safety metrics
        safety_metrics = self.calculate_safety_metrics()

        # Calculate robustness metrics
        robustness_metrics = self.calculate_robustness_metrics()

        # Generate reports
        performance_report = self.generate_performance_report(performance_metrics)
        accuracy_report = self.generate_accuracy_report(accuracy_metrics)
        safety_report = self.generate_safety_report(safety_metrics)

        # Publish reports
        self.publish_reports(
            performance_report, accuracy_report, safety_report,
            performance_metrics, accuracy_metrics, safety_metrics, robustness_metrics
        )

        # Log summary
        self.get_logger().info(
            f'Validation Report - Performance: {performance_metrics["avg_processing_time"]:.3f}s, '
            f'Accuracy: {accuracy_metrics["overall_accuracy"]:.3f}, '
            f'Safety: {safety_metrics["violation_rate"]:.3f}, '
            f'Robustness: {robustness_metrics["recovery_rate"]:.3f}'
        )

    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        metrics = {
            'avg_processing_time': 0.0,
            'min_processing_time': 0.0,
            'max_processing_time': 0.0,
            'std_processing_time': 0.0,
            'throughput': 0.0,
            'timing_stability': 0.0
        }

        if self.timing_buffer:
            times = list(self.timing_buffer)
            metrics.update({
                'avg_processing_time': statistics.mean(times),
                'min_processing_time': min(times),
                'max_processing_time': max(times),
                'std_processing_time': statistics.stdev(times) if len(times) > 1 else 0.0,
                'throughput': len(times) / (time.time() - self.start_time) if len(times) > 0 else 0.0,
                'timing_stability': 1.0 - (metrics['std_processing_time'] / (metrics['avg_processing_time'] + 1e-6))
            })

        return metrics

    def calculate_accuracy_metrics(self):
        """Calculate accuracy metrics"""
        metrics = {
            'overall_accuracy': 0.0,
            'action_success_rate': 0.0,
            'perception_accuracy': 0.0,
            'command_understanding_rate': 0.0
        }

        if self.action_success_buffer:
            action_success_rate = statistics.mean(self.action_success_buffer)
            metrics['action_success_rate'] = action_success_rate

        if self.perception_accuracy_buffer:
            perception_accuracy = statistics.mean(self.perception_accuracy_buffer)
            metrics['perception_accuracy'] = perception_accuracy

        # Overall accuracy as combination of action and perception
        if self.action_success_buffer and self.perception_accuracy_buffer:
            metrics['overall_accuracy'] = (
                metrics['action_success_rate'] * 0.6 +
                metrics['perception_accuracy'] * 0.4
            )

        # Command understanding rate (simplified)
        if self.total_evaluations > 0:
            metrics['command_understanding_rate'] = (
                self.successful_evaluations / self.total_evaluations
            )

        return metrics

    def calculate_safety_metrics(self):
        """Calculate safety metrics"""
        total_time = time.time() - self.start_time
        metrics = {
            'violation_rate': 0.0,
            'emergency_stop_rate': 0.0,
            'average_response_time': 0.0,
            'safety_score': 0.0
        }

        if total_time > 0:
            metrics['violation_rate'] = self.safety_violations / total_time * 60  # Per minute
            metrics['emergency_stop_rate'] = self.emergency_stops / total_time * 60  # Per minute

        # Safety score (higher is better, fewer violations = higher score)
        metrics['safety_score'] = max(0.0, 1.0 - (metrics['violation_rate'] / 10.0))  # Arbitrary scaling

        return metrics

    def calculate_robustness_metrics(self):
        """Calculate robustness metrics"""
        total_time = time.time() - self.start_time
        metrics = {
            'recovery_rate': 0.0,
            'uptime_percentage': 0.0,
            'error_frequency': 0.0,
            'robustness_score': 0.0
        }

        if total_time > 0:
            metrics['recovery_rate'] = self.error_recoveries / total_time * 60  # Per minute
            metrics['error_frequency'] = (
                (self.total_evaluations - self.successful_evaluations) / total_time * 60
            )

        # Robustness score (higher is better)
        if self.total_evaluations > 0:
            success_rate = self.successful_evaluations / self.total_evaluations
            metrics['robustness_score'] = success_rate

        return metrics

    def generate_performance_report(self, metrics):
        """Generate performance-focused report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'performance',
            'metrics': {
                'avg_processing_time_ms': metrics['avg_processing_time'] * 1000,
                'min_processing_time_ms': metrics['min_processing_time'] * 1000,
                'max_processing_time_ms': metrics['max_processing_time'] * 1000,
                'std_processing_time_ms': metrics['std_processing_time'] * 1000,
                'throughput_hz': metrics['throughput'],
                'timing_stability': metrics['timing_stability'],
                'sample_count': len(self.timing_buffer)
            }
        }

    def generate_accuracy_report(self, metrics):
        """Generate accuracy-focused report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'accuracy',
            'metrics': {
                'overall_accuracy': metrics['overall_accuracy'],
                'action_success_rate': metrics['action_success_rate'],
                'perception_accuracy': metrics['perception_accuracy'],
                'command_understanding_rate': metrics['command_understanding_rate'],
                'action_sample_count': len(self.action_success_buffer),
                'perception_sample_count': len(self.perception_accuracy_buffer)
            }
        }

    def generate_safety_report(self, metrics):
        """Generate safety-focused report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'safety',
            'metrics': {
                'violation_rate_per_minute': metrics['violation_rate'],
                'emergency_stop_rate_per_minute': metrics['emergency_stop_rate'],
                'safety_score': metrics['safety_score'],
                'total_violations': self.safety_violations,
                'total_emergency_stops': self.emergency_stops
            }
        }

    def publish_reports(self, performance_report, accuracy_report, safety_report,
                       performance_metrics, accuracy_metrics, safety_metrics, robustness_metrics):
        """Publish validation reports"""
        # Publish performance report
        perf_msg = String()
        perf_msg.data = json.dumps(performance_report)
        self.performance_pub.publish(perf_msg)

        # Publish accuracy report
        acc_msg = String()
        acc_msg.data = json.dumps(accuracy_report)
        self.accuracy_pub.publish(acc_msg)

        # Publish safety report
        safety_msg = String()
        safety_msg.data = json.dumps(safety_report)
        self.safety_pub.publish(safety_msg)

        # Publish comprehensive report
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'operational',
            'performance': performance_metrics,
            'accuracy': accuracy_metrics,
            'safety': safety_metrics,
            'robustness': robustness_metrics,
            'totals': {
                'total_evaluations': self.total_evaluations,
                'successful_evaluations': self.successful_evaluations,
                'failed_evaluations': self.failed_evaluations,
                'safety_violations': self.safety_violations,
                'emergency_stops': self.emergency_stops,
                'error_recoveries': self.error_recoveries,
                'uptime_seconds': time.time() - self.start_time
            }
        }

        comp_msg = String()
        comp_msg.data = json.dumps(comprehensive_report)
        self.comprehensive_pub.publish(comp_msg)

def main(args=None):
    rclpy.init(args=args)
    metrics_node = VLAValidationMetricsNode()

    try:
        rclpy.spin(metrics_node)
    except KeyboardInterrupt:
        metrics_node.get_logger().info('VLA validation metrics system shutting down')
    finally:
        metrics_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for VLA Evaluation and Validation

Key best practices for evaluating and validating VLA systems:

1. **Multi-dimensional Assessment**: Evaluate perception, language understanding, and action execution separately and together
2. **Real-time Performance**: Monitor processing times and ensure real-time constraints are met
3. **Safety Compliance**: Continuously validate that actions meet safety requirements
4. **Robustness Testing**: Test system behavior under various environmental conditions
5. **Uncertainty Quantification**: Assess and report system confidence in its decisions
6. **Domain Transfer**: Validate simulation-to-real transfer capabilities
7. **Continuous Monitoring**: Implement ongoing performance monitoring
8. **Reproducible Experiments**: Ensure evaluations are reproducible and comparable
9. **Human-in-the-Loop**: Include human evaluation for natural language interactions
10. **Comprehensive Reporting**: Provide detailed metrics and reports for analysis

### Physical Grounding and Simulation-to-Real Mapping

When evaluating VLA systems:

- **Hardware Fidelity**: Ensure evaluation metrics account for real hardware capabilities
- **Environmental Conditions**: Test under various lighting, noise, and environmental conditions
- **Latency Considerations**: Account for processing delays in real-time systems
- **Safety Margins**: Maintain appropriate safety margins between simulation and real operation
- **Calibration Validation**: Verify that sensor calibrations are accurate in real environments
- **Performance Degradation**: Monitor for performance degradation between simulation and real systems
- **Resource Utilization**: Validate that real hardware can sustain required computational loads

### Troubleshooting Validation Issues

Common validation problems and solutions:

- **Performance Bottlenecks**: Profile each component separately to identify issues
- **Accuracy Degradation**: Analyze failure cases to understand systematic issues
- **Safety Violations**: Implement additional safety checks and validation layers
- **Robustness Issues**: Test with more diverse datasets and environmental conditions
- **Domain Gap Problems**: Implement domain adaptation techniques or improve simulation fidelity
- **Timing Violations**: Optimize algorithms or upgrade hardware resources

### Summary

This chapter covered evaluation and validation techniques for Vision-Language-Action (VLA) systems in robotics. You learned about implementing basic and advanced evaluation systems, simulation-to-real validation frameworks, and comprehensive metrics reporting. Proper evaluation and validation are essential for ensuring that VLA systems perform reliably, safely, and effectively in real-world applications. The validation systems help identify potential issues before deployment and provide continuous monitoring during operation. In the next chapter, we'll explore deployment and optimization strategies for VLA systems in production robotics applications.