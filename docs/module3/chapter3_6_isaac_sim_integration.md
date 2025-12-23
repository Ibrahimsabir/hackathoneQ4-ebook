# Module 3: AI Robot Brain – NVIDIA Isaac

## Chapter 3.6: Isaac Sim Integration

This chapter explores Isaac Sim integration with Isaac ROS, focusing on how to use Isaac Sim for advanced simulation, testing, and validation of AI-powered robotic systems. Isaac Sim provides a photorealistic simulation environment that leverages NVIDIA's rendering and physics capabilities.

### Understanding Isaac Sim

Isaac Sim is NVIDIA's advanced robotics simulation environment that provides:

- **Photorealistic Rendering**: High-fidelity graphics for realistic sensor simulation
- **PhysX Physics Engine**: Accurate physics simulation with multi-body dynamics
- **AI Training Environment**: Synthetic data generation for AI model training
- **Hardware Acceleration**: Optimized for NVIDIA GPUs and RTX technology
- **ROS 2 Integration**: Seamless integration with ROS 2 ecosystems
- **Digital Twin Capabilities**: Accurate representation of real-world systems

### Isaac Sim Architecture and Components

The Isaac Sim architecture includes:

```
+-------------------+
|   Isaac Sim       |
|   (Simulation)   |
+-------------------+
|   Physics Engine  |
|   (PhysX)        |
+-------------------+
|   Rendering       |
|   (RTX)          |
+-------------------+
|   Sensor          |
|   Simulation     |
+-------------------+
|   Isaac ROS       |
|   Bridge         |
+-------------------+
```

### Setting Up Isaac Sim

Installing and configuring Isaac Sim for robotics simulation:

```bash
# Install Isaac Sim
# Download from NVIDIA Developer portal
# Isaac Sim is available as Omniverse extension

# For Isaac ROS integration
sudo apt update
sudo apt install nvidia-isaac-sim

# Verify installation
isaac-sim --version
```

### Isaac Sim Python API for Robotics

Using Isaac Sim's Python API for robotics simulation:

```python
import carb
import omni
import omni.ext
import omni.usd
from pxr import UsdGeom, Gf, Usd, Sdf
import numpy as np

class IsaacSimRobotManager:
    def __init__(self):
        self.stage = None
        self.robot_prim = None
        self.simulation_context = None

    def setup_scene(self):
        """Set up Isaac Sim scene"""
        # Get the USD stage
        self.stage = omni.usd.get_context().get_stage()

        # Create a simple ground plane
        self.create_ground_plane()

        # Create lighting
        self.create_lighting()

        # Set up physics scene
        self.setup_physics()

    def create_ground_plane(self):
        """Create a ground plane for the simulation"""
        # Create ground plane
        plane_path = Sdf.Path("/World/GroundPlane")
        plane_geom = UsdGeom.Mesh.Define(self.stage, plane_path)

        # Set plane properties
        plane_geom.CreatePointsAttr([(0, 0, 0), (10, 0, 0), (10, 0, 10), (0, 0, 10)])
        plane_geom.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        plane_geom.CreateFaceVertexCountsAttr([4])
        plane_geom.CreateExtentAttr([(-5, -5, 0), (5, 5, 0)])

    def create_lighting(self):
        """Create lighting in the simulation"""
        # Create dome light
        dome_light_path = Sdf.Path("/World/DomeLight")
        dome_light = UsdGeom.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr(300)

    def setup_physics(self):
        """Set up physics simulation"""
        # Create physics scene
        scene_path = Sdf.Path("/World/PhysicsScene")
        physics_scene = UsdPhysics.Scene.Define(self.stage, scene_path)

        # Set gravity
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    def spawn_robot(self, robot_usd_path, position=(0, 0, 0)):
        """Spawn a robot in the simulation"""
        # Import robot from USD file
        robot_path = Sdf.Path("/World/Robot")

        # Add reference to robot USD
        self.stage.GetRootLayer().subLayerPaths.append(robot_usd_path)

        # Set initial position
        robot_prim = self.stage.GetPrimAtPath(robot_path)
        if robot_prim:
            xform_api = UsdGeom.Xformable(robot_prim)
            xform_api.AddTranslateOp().Set(Gf.Vec3f(*position))

        self.robot_prim = robot_prim
        return robot_prim

    def add_sensors(self, robot_path):
        """Add sensors to the robot"""
        # Add camera sensor
        camera_path = Sdf.Path(f"{robot_path}/Camera")
        camera = UsdGeom.Camera.Define(self.stage, camera_path)

        # Set camera properties
        camera.GetFocalLengthAttr().Set(24.0)
        camera.GetHorizontalApertureAttr().Set(36.0)
        camera.GetVerticalApertureAttr().Set(20.25)

        # Add IMU sensor (conceptual - actual implementation would vary)
        imu_path = Sdf.Path(f"{robot_path}/Imu")
        # IMU implementation would depend on Isaac Sim extensions

    def run_simulation(self, steps=1000):
        """Run the simulation for specified steps"""
        # Initialize simulation context
        self.simulation_context = omni.physics.get_simulation_context()

        # Step through simulation
        for i in range(steps):
            self.simulation_context.step(render=True)

            # Optional: Process sensor data or robot control
            if i % 100 == 0:
                carb.log_info(f"Simulation step {i}")

    def cleanup(self):
        """Clean up simulation resources"""
        if self.simulation_context:
            self.simulation_context.close()
```

### Isaac Sim ROS Bridge Configuration

Configuring the Isaac Sim to ROS bridge for robotics applications:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np

class IsaacSimRosBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # Publishers for sensor data
        self.camera_pub = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        self.lidar_pub = self.create_publisher(
            LaserScan,
            '/scan',
            10
        )

        self.imu_pub = self.create_publisher(
            Imu,
            '/imu/data',
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        # Subscribers for robot commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.get_logger().info('Isaac Sim ROS bridge initialized')

    def cmd_vel_callback(self, msg):
        """Process velocity commands from ROS"""
        # In Isaac Sim, this would send commands to the simulated robot
        # For demonstration, we'll just log the command
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')

    def publish_camera_data(self, camera_data):
        """Publish camera data from Isaac Sim"""
        image_msg = Image()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera_frame'
        image_msg.width = camera_data.shape[1]
        image_msg.height = camera_data.shape[0]
        image_msg.encoding = 'rgb8'
        image_msg.is_bigendian = False
        image_msg.step = camera_data.shape[1] * 3  # 3 bytes per pixel (RGB)
        image_msg.data = camera_data.tobytes()

        self.camera_pub.publish(image_msg)

    def publish_lidar_data(self, lidar_data):
        """Publish LiDAR data from Isaac Sim"""
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / len(lidar_data)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.0
        scan_msg.range_min = 0.1
        scan_msg.range_max = 30.0
        scan_msg.ranges = lidar_data.tolist()

        self.lidar_pub.publish(scan_msg)

    def publish_imu_data(self, linear_accel, angular_vel, orientation):
        """Publish IMU data from Isaac Sim"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_frame'
        imu_msg.linear_acceleration.x = linear_accel[0]
        imu_msg.linear_acceleration.y = linear_accel[1]
        imu_msg.linear_acceleration.z = linear_accel[2]
        imu_msg.angular_velocity.x = angular_vel[0]
        imu_msg.angular_velocity.y = angular_vel[1]
        imu_msg.angular_velocity.z = angular_vel[2]
        imu_msg.orientation.x = orientation[0]
        imu_msg.orientation.y = orientation[1]
        imu_msg.orientation.z = orientation[2]
        imu_msg.orientation.w = orientation[3]

        self.imu_pub.publish(imu_msg)

    def publish_odometry(self, position, orientation, linear_vel, angular_vel):
        """Publish odometry data from Isaac Sim"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]
        odom_msg.twist.twist.linear.x = linear_vel[0]
        odom_msg.twist.twist.linear.y = linear_vel[1]
        odom_msg.twist.twist.linear.z = linear_vel[2]
        odom_msg.twist.twist.angular.x = angular_vel[0]
        odom_msg.twist.twist.angular.y = angular_vel[1]
        odom_msg.twist.twist.angular.z = angular_vel[2]

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimRosBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info('Isaac Sim ROS bridge shutting down')
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac Sim for AI Training

Using Isaac Sim for AI model training and synthetic data generation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import json
import os
from datetime import datetime

class IsaacSimAITraining(Node):
    def __init__(self):
        super().__init__('isaac_sim_ai_training')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/ai/detections',
            self.detection_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/training_status',
            10
        )

        # Training data parameters
        self.training_data_dir = '/tmp/isaac_sim_training_data'
        self.annotation_format = 'coco'  # COCO, Pascal VOC, or custom
        self.data_counter = 0
        self.max_training_samples = 10000

        # Create training data directory
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training_data_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.training_data_dir, 'annotations'), exist_ok=True)

        # Current frame and annotations
        self.current_image = None
        self.current_detections = None
        self.camera_info = None

        # Training statistics
        self.stats = {
            'total_samples': 0,
            'samples_with_objects': 0,
            'object_counts': {}
        }

        self.get_logger().info('Isaac Sim AI training initialized')

    def image_callback(self, msg):
        """Process camera image for training data"""
        self.current_image = msg

        # Save training data if we have detections
        if self.current_detections is not None:
            self.save_training_sample()

    def detection_callback(self, msg):
        """Process detections for training"""
        self.current_detections = msg

    def camera_info_callback(self, msg):
        """Process camera calibration info"""
        self.camera_info = msg

    def save_training_sample(self):
        """Save current frame and annotations for training"""
        if self.current_image is None or self.current_detections is None:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")

            # Create annotation data
            annotation_data = self.create_annotation_data()

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"image_{timestamp}_{self.data_counter:06d}.jpg"
            annotation_filename = f"annotation_{timestamp}_{self.data_counter:06d}.json"

            # Save image
            image_path = os.path.join(self.training_data_dir, 'images', image_filename)
            # In a real implementation, you'd use cv2.imwrite(image_path, cv_image)

            # Save annotation
            annotation_path = os.path.join(self.training_data_dir, 'annotations', annotation_filename)
            with open(annotation_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)

            # Update statistics
            self.data_counter += 1
            self.stats['total_samples'] += 1

            if len(self.current_detections.detections) > 0:
                self.stats['samples_with_objects'] += 1

                # Count object classes
                for detection in self.current_detections.detections:
                    for result in detection.results:
                        class_id = result.hypothesis.class_id
                        if class_id in self.stats['object_counts']:
                            self.stats['object_counts'][class_id] += 1
                        else:
                            self.stats['object_counts'][class_id] = 1

            # Log progress
            self.get_logger().info(
                f'Saved training sample {self.data_counter}/{self.max_training_samples}: '
                f'{image_filename} with {len(self.current_detections.detections)} objects'
            )

            # Publish status
            status_msg = String()
            status_msg.data = (
                f'Training samples: {self.data_counter}, '
                f'With objects: {self.stats["samples_with_objects"]}, '
                f'Classes: {list(self.stats["object_counts"].keys())}'
            )
            self.status_pub.publish(status_msg)

            # Check if we've reached maximum samples
            if self.data_counter >= self.max_training_samples:
                self.get_logger().info('Maximum training samples reached')
                self.generate_training_summary()

        except Exception as e:
            self.get_logger().error(f'Error saving training sample: {str(e)}')

    def create_annotation_data(self):
        """Create annotation data in COCO format"""
        # In Isaac Sim, this would use the built-in annotation tools
        # For demonstration, we'll create a simplified COCO-style annotation

        annotation = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "Synthetic training data from Isaac Sim",
                "contributor": "Isaac Sim AI Training",
                "date_created": datetime.now().isoformat()
            },
            "images": [{
                "id": self.data_counter,
                "width": self.current_image.width,
                "height": self.current_image.height,
                "file_name": f"image_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{self.data_counter:06d}.jpg",
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat()
            }],
            "annotations": [],
            "categories": []
        }

        # Add object annotations
        category_names = set()
        for i, detection in enumerate(self.current_detections.detections):
            bbox = detection.bbox
            category_names.add(detection.results[0].hypothesis.class_id if detection.results else 'unknown')

            annotation["annotations"].append({
                "id": i,
                "image_id": self.data_counter,
                "category_id": list(category_names).index(detection.results[0].hypothesis.class_id) if detection.results else 0,
                "segmentation": [],  # Could add segmentation masks
                "area": bbox.size_x * bbox.size_y,
                "bbox": [
                    bbox.center.x - bbox.size_x / 2,  # x_min
                    bbox.center.y - bbox.size_y / 2,  # y_min
                    bbox.size_x,                      # width
                    bbox.size_y                       # height
                ],
                "iscrowd": 0,
                "score": detection.results[0].hypothesis.score if detection.results else 0.0
            })

        # Add categories
        for i, name in enumerate(category_names):
            annotation["categories"].append({
                "id": i,
                "name": name,
                "supercategory": "object"
            })

        return annotation

    def generate_training_summary(self):
        """Generate summary of collected training data"""
        summary = {
            "dataset_info": {
                "total_samples": self.stats['total_samples'],
                "samples_with_objects": self.stats['samples_with_objects'],
                "object_distribution": self.stats['object_counts'],
                "collection_date": datetime.now().isoformat()
            }
        }

        summary_path = os.path.join(self.training_data_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(f'Training summary saved to {summary_path}')

def main(args=None):
    rclpy.init(args=args)
    trainer = IsaacSimAITraining()

    try:
        rclpy.spin(trainer)
    except KeyboardInterrupt:
        trainer.get_logger().info('AI training data collection shutting down')
    finally:
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac Sim for Perception Testing

Testing perception algorithms in Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import Detection2DArray, Classification2D
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacSimPerceptionTester(Node):
    def __init__(self):
        super().__init__('isaac_sim_perception_tester')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.camera_sub = self.create_subscription(
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

        self.gt_detections_sub = self.create_subscription(
            Detection2DArray,
            '/ground_truth/detections',
            self.gt_detections_callback,
            10
        )

        self.perception_detections_sub = self.create_subscription(
            Detection2DArray,
            '/perception/detections',
            self.perception_detections_callback,
            10
        )

        # Publishers for test results
        self.accuracy_pub = self.create_publisher(
            Float32,
            '/perception/accuracy',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/perception/test_status',
            10
        )

        # Test parameters
        self.iou_threshold = 0.5  # Intersection over Union threshold
        self.confidence_threshold = 0.5
        self.test_results = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_detections': 0,
            'total_gt_objects': 0
        }

        # Storage for current data
        self.current_image = None
        self.current_gt_detections = None
        self.current_perception_detections = None
        self.current_scan = None

        # Test timer
        self.test_timer = self.create_timer(1.0, self.run_perception_test)

        self.get_logger().info('Isaac Sim perception tester initialized')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def gt_detections_callback(self, msg):
        """Process ground truth detections"""
        self.current_gt_detections = msg

    def perception_detections_callback(self, msg):
        """Process perception algorithm detections"""
        self.current_perception_detections = msg

    def run_perception_test(self):
        """Run perception accuracy test"""
        if (self.current_gt_detections is None or
            self.current_perception_detections is None):
            return

        # Calculate accuracy metrics
        accuracy = self.calculate_detection_accuracy()

        # Publish accuracy
        acc_msg = Float32()
        acc_msg.data = accuracy
        self.accuracy_pub.publish(acc_msg)

        # Publish status
        status_msg = String()
        status_msg.data = (
            f'Accuracy: {accuracy:.3f}, '
            f'TP: {self.test_results["true_positives"]}, '
            f'FP: {self.test_results["false_positives"]}, '
            f'FN: {self.test_results["false_negatives"]}'
        )
        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f'Perception Test - Accuracy: {accuracy:.3f}, '
            f'True Positives: {self.test_results["true_positives"]}, '
            f'False Positives: {self.test_results["false_positives"]}, '
            f'False Negatives: {self.test_results["false_negatives"]}'
        )

    def calculate_detection_accuracy(self):
        """Calculate detection accuracy metrics"""
        # Reset counters for this test
        tp = fp = fn = 0

        gt_detections = self.current_gt_detections.detections
        pred_detections = self.current_perception_detections.detections

        # Track matched ground truth objects
        gt_matched = [False] * len(gt_detections)

        # For each predicted detection, find the best matching ground truth
        for pred_det in pred_detections:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_det in enumerate(gt_detections):
                if not gt_matched[i]:  # Only consider unmatched ground truths
                    iou = self.calculate_iou(pred_det.bbox, gt_det.bbox)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_gt_idx = i

            if best_gt_idx != -1:
                # True positive: matched with ground truth
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive: no matching ground truth
                fp += 1

        # False negatives: ground truth objects that weren't detected
        fn = len(gt_detections) - sum(gt_matched)

        # Update cumulative results
        self.test_results['true_positives'] += tp
        self.test_results['false_positives'] += fp
        self.test_results['false_negatives'] += fn
        self.test_results['total_detections'] += len(pred_detections)
        self.test_results['total_gt_objects'] += len(gt_detections)

        # Calculate accuracy (precision and recall)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Extract coordinates
        x1_min = bbox1.center.x - bbox1.size_x / 2
        y1_min = bbox1.center.y - bbox1.size_y / 2
        x1_max = bbox1.center.x + bbox1.size_x / 2
        y1_max = bbox1.center.y + bbox1.size_y / 2

        x2_min = bbox2.center.x - bbox2.size_x / 2
        y2_min = bbox2.center.y - bbox2.size_y / 2
        x2_max = bbox2.center.x + bbox2.size_x / 2
        y2_max = bbox2.center.y + bbox2.size_y / 2

        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        else:
            inter_area = 0

        # Calculate union area
        area1 = bbox1.size_x * bbox1.size_y
        area2 = bbox2.size_x * bbox2.size_y
        union_area = area1 + area2 - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

def main(args=None):
    rclpy.init(args=args)
    tester = IsaacSimPerceptionTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Perception testing shutting down')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac Sim for Navigation Testing

Testing navigation algorithms in Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class IsaacSimNavigationTester(Node):
    def __init__(self):
        super().__init__('isaac_sim_navigation_tester')

        # Publishers and subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publishers for test results
        self.path_accuracy_pub = self.create_publisher(
            Float32,
            '/navigation/path_accuracy',
            10
        )

        self.success_rate_pub = self.create_publisher(
            Float32,
            '/navigation/success_rate',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/navigation/test_status',
            10
        )

        # Test parameters
        self.path_accuracy_threshold = 0.5  # meters
        self.goal_tolerance = 0.5  # meters
        self.collision_threshold = 0.3  # meters

        # Test state
        self.current_path = None
        self.current_pose = None
        self.current_scan = None
        self.current_map = None
        self.current_cmd = None
        self.start_pose = None
        self.goal_pose = None
        self.traveled_path = []
        self.test_active = False
        self.collision_occurred = False
        self.goal_reached = False

        # Test statistics
        self.test_stats = {
            'completed_tests': 0,
            'successful_navigations': 0,
            'average_path_accuracy': 0.0,
            'average_time': 0.0,
            'collision_rate': 0.0
        }

        # Control timer
        self.test_timer = self.create_timer(0.1, self.run_navigation_test)

        self.get_logger().info('Isaac Sim navigation tester initialized')

    def path_callback(self, msg):
        """Process navigation path"""
        self.current_path = msg
        if not self.test_active:
            # New test started
            self.start_new_test()

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose

        # Store traveled path
        if self.current_pose:
            self.traveled_path.append((
                self.current_pose.position.x,
                self.current_pose.position.y
            ))

    def scan_callback(self, msg):
        """Process laser scan for collision detection"""
        self.current_scan = msg

        # Check for collisions
        if self.current_scan:
            ranges = np.array(self.current_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]

            if len(valid_ranges) > 0 and np.min(valid_ranges) < self.collision_threshold:
                self.collision_occurred = True
                self.get_logger().warn('Collision detected during navigation!')

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.current_map = msg

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.current_cmd = msg

    def start_new_test(self):
        """Start a new navigation test"""
        if self.current_pose and self.current_path and len(self.current_path.poses) > 0:
            self.start_pose = self.current_pose
            self.goal_pose = self.current_path.poses[-1].pose
            self.traveled_path = [(self.start_pose.position.x, self.start_pose.position.y)]
            self.test_active = True
            self.collision_occurred = False
            self.goal_reached = False
            self.test_start_time = self.get_clock().now()

            self.get_logger().info(
                f'Navigation test started: '
                f'From ({self.start_pose.position.x:.2f}, {self.start_pose.position.y:.2f}) '
                f'To ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f})'
            )

    def run_navigation_test(self):
        """Run navigation test and evaluate performance"""
        if not self.test_active or not self.current_pose:
            return

        # Check if goal is reached
        if self.goal_pose:
            dx = self.current_pose.position.x - self.goal_pose.position.x
            dy = self.current_pose.position.y - self.goal_pose.position.y
            distance_to_goal = np.sqrt(dx*dx + dy*dy)

            if distance_to_goal <= self.goal_tolerance:
                self.goal_reached = True
                self.test_active = False

                # Calculate test results
                test_duration = (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9

                # Calculate path efficiency
                if self.current_path and len(self.current_path.poses) > 1:
                    # Calculate planned path length
                    planned_length = 0
                    for i in range(len(self.current_path.poses) - 1):
                        p1 = self.current_path.poses[i].pose.position
                        p2 = self.current_path.poses[i+1].pose.position
                        dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                        planned_length += dist

                    # Calculate actual path length
                    actual_length = 0
                    for i in range(len(self.traveled_path) - 1):
                        p1 = self.traveled_path[i]
                        p2 = self.traveled_path[i+1]
                        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        actual_length += dist

                    # Calculate path efficiency
                    path_efficiency = planned_length / actual_length if actual_length > 0 else 0

                    # Publish results
                    efficiency_msg = Float32()
                    efficiency_msg.data = path_efficiency
                    self.path_accuracy_pub.publish(efficiency_msg)

                # Update statistics
                self.test_stats['completed_tests'] += 1
                if not self.collision_occurred and self.goal_reached:
                    self.test_stats['successful_navigations'] += 1

                # Calculate success rate
                success_rate = self.test_stats['successful_navigations'] / self.test_stats['completed_tests'] if self.test_stats['completed_tests'] > 0 else 0

                success_msg = Float32()
                success_msg.data = success_rate
                self.success_rate_pub.publish(success_msg)

                # Publish status
                status_msg = String()
                status_msg.data = (
                    f'Test completed - Success: {self.goal_reached}, '
                    f'Collision: {self.collision_occurred}, '
                    f'Success Rate: {success_rate:.3f}, '
                    f'Time: {test_duration:.2f}s'
                )
                self.status_pub.publish(status_msg)

                self.get_logger().info(
                    f'Navigation test completed: '
                    f'Success: {self.goal_reached}, '
                    f'Collision: {self.collision_occurred}, '
                    f'Time: {test_duration:.2f}s, '
                    f'Success Rate: {success_rate:.3f}'
                )

    def calculate_path_accuracy(self):
        """Calculate path following accuracy"""
        if not self.current_path or not self.traveled_path:
            return 0.0

        # Calculate average deviation from planned path
        total_deviation = 0
        deviations_count = 0

        for actual_pos in self.traveled_path:
            min_distance_to_path = float('inf')

            # Find minimum distance to any point in the planned path
            for path_pose in self.current_path.poses:
                path_pos = (path_pose.pose.position.x, path_pose.pose.position.y)
                distance = np.sqrt((actual_pos[0] - path_pos[0])**2 + (actual_pos[1] - path_pos[1])**2)
                min_distance_to_path = min(min_distance_to_path, distance)

            if min_distance_to_path < float('inf'):
                total_deviation += min_distance_to_path
                deviations_count += 1

        average_deviation = total_deviation / deviations_count if deviations_count > 0 else 0
        return 1.0 / (1.0 + average_deviation)  # Convert to accuracy metric

def main(args=None):
    rclpy.init(args=args)
    nav_tester = IsaacSimNavigationTester()

    try:
        rclpy.spin(nav_tester)
    except KeyboardInterrupt:
        nav_tester.get_logger().info('Navigation testing shutting down')
    finally:
        nav_tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac Sim for Hardware-in-the-Loop Testing

Implementing hardware-in-the-loop testing with Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, JointState, Imu
from geometry_msgs.msg import Twist, WrenchStamped
from std_msgs.msg import Bool, String
from diagnostic_msgs.msg import DiagnosticArray
import numpy as np
import threading
import time

class IsaacSimHILTester(Node):
    def __init__(self):
        super().__init__('isaac_sim_hil_tester')

        # Publishers for simulated sensor data
        self.camera_pub = self.create_publisher(Image, '/simulated_camera/image_raw', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/simulated_scan', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/simulated_joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/simulated_imu/data', 10)
        self.force_torque_pub = self.create_publisher(WrenchStamped, '/simulated_force_torque', 10)

        # Subscribers for real robot commands
        self.cmd_vel_sub = self.create_subscription(Twist, '/real_cmd_vel', self.real_cmd_callback, 10)
        self.joint_cmd_sub = self.create_subscription(JointState, '/real_joint_commands', self.joint_cmd_callback, 10)

        # Publishers for diagnostics
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.sync_status_pub = self.create_publisher(Bool, '/hil_sync_status', 10)

        # HIL parameters
        self.simulation_speed = 1.0  # Real-time simulation
        self.latency_compensation = 0.05  # 50ms compensation
        self.sync_tolerance = 0.1  # 100ms tolerance

        # Robot state
        self.real_robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'linear_vel': np.array([0.0, 0.0, 0.0]),
            'angular_vel': np.array([0.0, 0.0, 0.0]),
            'joint_positions': {},
            'joint_velocities': {},
            'joint_efforts': {}
        }

        self.simulated_robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
            'linear_vel': np.array([0.0, 0.0, 0.0]),
            'angular_vel': np.array([0.0, 0.0, 0.0]),
            'joint_positions': {},
            'joint_velocities': {},
            'joint_efforts': {}
        }

        # Synchronization state
        self.sync_active = False
        self.last_sync_time = time.time()

        # Control timer
        self.hil_timer = self.create_timer(0.01, self.hil_control_loop)  # 100Hz

        self.get_logger().info('Isaac Sim HIL tester initialized')

    def real_cmd_callback(self, msg):
        """Process real robot velocity commands"""
        # Apply commands to simulated robot
        self.apply_velocity_command(msg)

    def joint_cmd_callback(self, msg):
        """Process real robot joint commands"""
        # Apply joint commands to simulated robot
        self.apply_joint_commands(msg)

    def hil_control_loop(self):
        """Main HIL control loop"""
        current_time = time.time()

        # Update simulated robot based on commands
        self.update_simulated_robot()

        # Publish simulated sensor data
        self.publish_simulated_sensors()

        # Check synchronization
        self.check_synchronization()

        # Publish diagnostics
        self.publish_diagnostics()

        # Update last sync time
        self.last_sync_time = current_time

    def apply_velocity_command(self, cmd_msg):
        """Apply velocity command to simulated robot"""
        # Update simulated robot state based on velocity command
        dt = 0.01  # Assuming 100Hz loop

        # Simple kinematic model (in a real implementation, this would be more complex)
        self.simulated_robot_state['linear_vel'][0] = cmd_msg.linear.x
        self.simulated_robot_state['angular_vel'][2] = cmd_msg.angular.z

        # Update position based on velocity
        self.simulated_robot_state['position'][0] += cmd_msg.linear.x * dt
        self.simulated_robot_state['position'][1] += cmd_msg.linear.y * dt
        # Update orientation based on angular velocity
        # (simplified - in reality, you'd integrate properly)

    def apply_joint_commands(self, joint_msg):
        """Apply joint commands to simulated robot"""
        # Update simulated joint positions based on commands
        for i, name in enumerate(joint_msg.name):
            if i < len(joint_msg.position):
                self.simulated_robot_state['joint_positions'][name] = joint_msg.position[i]

    def update_simulated_robot(self):
        """Update simulated robot state"""
        # In Isaac Sim, this would update the robot model in the simulation
        # For demonstration, we'll just log the state
        self.get_logger().debug(
            f'Simulated Robot - Position: {self.simulated_robot_state["position"]}, '
            f'Linear Vel: {self.simulated_robot_state["linear_vel"]}'
        )

    def publish_simulated_sensors(self):
        """Publish simulated sensor data"""
        # Publish simulated camera image
        self.publish_simulated_camera()

        # Publish simulated LiDAR
        self.publish_simulated_lidar()

        # Publish simulated joint states
        self.publish_simulated_joint_states()

        # Publish simulated IMU
        self.publish_simulated_imu()

    def publish_simulated_camera(self):
        """Publish simulated camera data"""
        # In Isaac Sim, this would capture image from simulated camera
        # For demonstration, we'll create a dummy image
        from sensor_msgs.msg import Image
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'simulated_camera'
        img_msg.width = 640
        img_msg.height = 480
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = 640 * 3  # 3 bytes per pixel
        img_msg.data = [0] * (640 * 480 * 3)  # Dummy data

        self.camera_pub.publish(img_msg)

    def publish_simulated_lidar(self):
        """Publish simulated LiDAR data"""
        from sensor_msgs.msg import LaserScan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'simulated_laser'
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / 360
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.0
        scan_msg.range_min = 0.1
        scan_msg.range_max = 30.0

        # Simulate some range data
        ranges = [2.0 + 0.5 * np.sin(i * np.pi / 180) for i in range(360)]
        scan_msg.ranges = ranges

        self.lidar_pub.publish(scan_msg)

    def publish_simulated_joint_states(self):
        """Publish simulated joint states"""
        from sensor_msgs.msg import JointState
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(self.simulated_robot_state['joint_positions'].keys())
        joint_msg.position = list(self.simulated_robot_state['joint_positions'].values())
        joint_msg.velocity = list(self.simulated_robot_state['joint_velocities'].values())
        joint_msg.effort = list(self.simulated_robot_state['joint_efforts'].values())

        self.joint_state_pub.publish(joint_msg)

    def publish_simulated_imu(self):
        """Publish simulated IMU data"""
        from geometry_msgs.msg import Vector3
        from geometry_msgs.msg import Quaternion
        from sensor_msgs.msg import Imu

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'simulated_imu'

        # Simulate IMU data
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 9.81  # Gravity
        imu_msg.angular_velocity.z = self.simulated_robot_state['angular_vel'][2]
        imu_msg.orientation.w = 1.0  # Simplified orientation

        self.imu_pub.publish(imu_msg)

    def check_synchronization(self):
        """Check HIL synchronization status"""
        current_time = time.time()
        time_since_sync = current_time - self.last_sync_time

        sync_status = Bool()
        sync_status.data = time_since_sync < self.sync_tolerance
        self.sync_status_pub.publish(sync_status)

        if not sync_status.data:
            self.get_logger().warn(f'HIL synchronization lost! Delay: {time_since_sync:.3f}s')

    def publish_diagnostics(self):
        """Publish HIL diagnostics"""
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Create diagnostic status
        status = DiagnosticStatus()
        status.name = 'Isaac Sim HIL Tester'
        status.level = DiagnosticStatus.OK
        status.message = 'HIL testing active'

        # Add key-value pairs for status details
        status.values.extend([
            KeyValue(key='Simulation Speed', value=f'{self.simulation_speed}x'),
            KeyValue(key='Latency Compensation', value=f'{self.latency_compensation}s'),
            KeyValue(key='Synchronization Status', value=str(self.sync_active)),
            KeyValue(key='Last Sync Time', value=f'{self.last_sync_time}')
        ])

        diag_array.status.append(status)
        self.diagnostic_pub.publish(diag_array)

def main(args=None):
    rclpy.init(args=args)
    hil_tester = IsaacSimHILTester()

    try:
        rclpy.spin(hil_tester)
    except KeyboardInterrupt:
        hil_tester.get_logger().info('HIL testing shutting down')
    finally:
        hil_tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Isaac Sim Integration

1. **Scene Optimization**: Optimize simulation scenes for performance while maintaining fidelity
2. **Sensor Accuracy**: Calibrate simulated sensors to match real hardware characteristics
3. **Physics Tuning**: Tune physics parameters to match real-world behavior
4. **Validation**: Validate simulation results against real-world data
5. **Performance Monitoring**: Monitor simulation performance and adjust as needed
6. **Deterministic Simulation**: Ensure reproducible results for testing
7. **Asset Quality**: Use high-quality 3D assets for realistic simulation

### Physical Grounding and Simulation-to-Real Mapping

When using Isaac Sim for robotics development:

- **Sensor Modeling**: Accurately model sensor noise and characteristics
- **Physics Parameters**: Calibrate physics parameters to match real hardware
- **Timing**: Consider simulation vs. real-time execution differences
- **Environmental Conditions**: Account for lighting, texture, and material properties
- **Validation**: Regularly validate simulation results against real-world performance
- **Transfer Learning**: Use domain randomization to improve sim-to-real transfer

### Troubleshooting Isaac Sim Issues

Common Isaac Sim problems and solutions:

- **Performance Issues**: Optimize scene complexity and graphics settings
- **Physics Instability**: Adjust physics parameters and timestep
- **Sensor Mismatch**: Calibrate sensor models to match real hardware
- **Synchronization Problems**: Check timing and communication between systems
- **Asset Loading**: Verify 3D assets are properly formatted and accessible

### Summary

This chapter covered Isaac Sim integration with Isaac ROS, focusing on how to use Isaac Sim for advanced simulation, testing, and validation of AI-powered robotic systems. You learned about Isaac Sim's architecture, how to configure the ROS bridge, how to use Isaac Sim for AI training and testing, and how to implement hardware-in-the-loop testing. Isaac Sim provides a powerful platform for developing and validating robotics applications in a safe, controlled, and photorealistic environment. The integration with Isaac ROS enables seamless testing and validation of robotic systems before deployment on real hardware. As we move to the next module, we'll explore Vision-Language-Action systems for multimodal AI in robotics.