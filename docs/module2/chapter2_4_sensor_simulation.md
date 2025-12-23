# Module 2: Digital Twins – Gazebo & Unity

## Chapter 2.4: Sensor Simulation and Integration

This chapter focuses on simulating various types of sensors in Gazebo and integrating them with ROS 2, which is crucial for creating realistic digital twins of robotic systems.

### Understanding Sensor Simulation in Gazebo

Sensor simulation in Gazebo provides realistic data that mimics real-world sensors, allowing developers to test perception algorithms, navigation systems, and other sensor-dependent functionalities in a controlled environment. The simulated sensors produce data that follows the same ROS message types as their real-world counterparts.

### Camera Sensors

Camera sensors are among the most commonly used sensors in robotics applications. Here's how to configure a camera sensor in Gazebo:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>camera</namespace>
        <remapping>~/out:=image_raw</remapping>
      </ros>
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

#### Camera Sensor Parameters Explained:
- **horizontal_fov**: Horizontal field of view in radians
- **image**: Resolution and format of the camera output
- **clip**: Near and far clipping distances
- **noise**: Simulated sensor noise to make data more realistic
- **frame_name**: TF frame for the camera

### Depth Camera Sensors

Depth cameras provide both color and depth information:

```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="depth">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>camera</namespace>
      </ros>
      <frame_name>camera_depth_optical_frame</frame_name>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR (Laser) Sensors

LiDAR sensors are essential for navigation and mapping:

```xml
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
          <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>laser</namespace>
        <remapping>scan:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

#### 3D LiDAR Configuration

For 3D LiDAR sensors like Velodyne:

```xml
<gazebo reference="velodyne_link">
  <sensor name="velodyne" type="ray">
    <always_on>true</always_on>
    <visualize>false</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle> <!-- -30 degrees -->
          <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
      <ros>
        <namespace>velodyne</namespace>
      </ros>
      <frame_name>velodyne_frame</frame_name>
      <min_range>0.1</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.01</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### IMU (Inertial Measurement Unit) Sensors

IMU sensors provide orientation, velocity, and acceleration data:

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>imu:=data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
      <gaussian_noise>0.01</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### GPS Sensors

For outdoor robots, GPS simulation is important:

```xml
<gazebo reference="gps_link">
  <sensor name="gps_sensor" type="gps">
    <always_on>true</always_on>
    <update_rate>1</update_rate>
    <plugin name="gps_plugin" filename="libgazebo_ros_gps.so">
      <ros>
        <namespace>gps</namespace>
      </ros>
      <frame_name>gps_link</frame_name>
      <update_rate>1.0</update_rate>
      <fix_topic>fix</fix_topic>
      <gaussian_noise>0.1</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### Multi-Sensor Integration Example

Here's a complete example showing how to integrate multiple sensors on a robot:

```xml
<?xml version="1.0"?>
<robot name="sensor_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Camera mount -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Laser mount -->
  <link name="laser_link">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- IMU mount -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Joints connecting sensors -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
  </joint>

  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0.1 0 0.05" rpy="0 0 0"/>
  </joint>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo sensor definitions -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100.0</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>camera</namespace>
          <remapping>~/out:=image_raw</remapping>
        </ros>
        <frame_name>camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="laser_link">
    <sensor name="laser" type="ray">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>laser</namespace>
          <remapping>scan:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>laser_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <ros>
          <namespace>imu</namespace>
        </ros>
        <frame_name>imu_link</frame_name>
        <initial_orientation_as_reference>false</initial_orientation_as_reference>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Sensor Data Processing in ROS 2

Once sensors are configured, you can process their data in ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers for different sensor types
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/laser/scan',
            self.laser_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.get_logger().info('Sensor processor initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process the image (example: detect edges)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Log information
            self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def laser_callback(self, msg):
        # Process laser scan data
        ranges = np.array(msg.ranges)

        # Remove invalid ranges (inf or nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Min obstacle distance: {min_distance:.2f}m')

        # Log some information
        self.get_logger().info(f'Laser scan: {len(msg.ranges)} points, '
                              f'range: {msg.range_min:.2f}-{msg.range_max:.2f}m')

    def imu_callback(self, msg):
        # Process IMU data
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        self.get_logger().info(f'IMU - Orientation: ({orientation.x:.2f}, '
                              f'{orientation.y:.2f}, {orientation.z:.2f}, {orientation.w:.2f})')

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down sensor processor')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Fusion and Integration

For complex applications, you may need to combine data from multiple sensors:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Initialize data storage
        self.laser_data = None
        self.imu_data = None
        self.last_cmd_time = self.get_clock().now()

        # Create subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/laser/scan',
            self.laser_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for sensor fusion
        self.timer = self.create_timer(0.1, self.fusion_callback)

    def laser_callback(self, msg):
        self.laser_data = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def fusion_callback(self):
        if self.laser_data is None or self.imu_data is None:
            return

        # Analyze laser data to detect obstacles
        ranges = np.array(self.laser_data.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)

            # Create velocity command based on sensor fusion
            cmd = Twist()

            if min_distance < 1.0:  # Obstacle within 1 meter
                # Stop and turn to avoid obstacle
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
            else:
                # Move forward
                cmd.linear.x = 0.5
                cmd.angular.z = 0.0

            # Apply IMU data for more sophisticated control
            if abs(self.imu_data.angular_velocity.z) > 0.1:
                # Compensate for rotation
                cmd.angular.z *= 0.8

            # Publish command
            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()
    rclpy.spin(fusion_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Noise and Realism

Adding realistic noise to sensors makes the simulation more accurate:

```xml
<sensor name="realistic_camera" type="camera">
  <camera name="head">
    <!-- ... other camera parameters ... -->
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>  <!-- Add realistic noise -->
    </noise>
  </camera>
</sensor>

<sensor name="realistic_lidar" type="ray">
  <ray>
    <!-- ... other ray parameters ... -->
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <gaussian_noise>0.01</gaussian_noise>  <!-- 1cm noise -->
    <!-- ... other parameters ... -->
  </plugin>
</sensor>
```

### Sensor Calibration and Validation

Validating sensor simulation against real-world data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/laser/scan',
            self.scan_callback,
            10
        )

        self.scan_history = []
        self.max_history = 100

    def scan_callback(self, msg):
        # Store scan data for analysis
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            # Calculate statistics
            mean_range = np.mean(valid_ranges)
            std_range = np.std(valid_ranges)

            # Log statistics
            self.get_logger().info(
                f'Scan stats - Mean: {mean_range:.2f}m, '
                f'Std: {std_range:.2f}m, '
                f'Min: {np.min(valid_ranges):.2f}m, '
                f'Max: {np.max(valid_ranges):.2f}m'
            )

            # Store for history analysis
            self.scan_history.append({
                'mean': mean_range,
                'std': std_range,
                'min': np.min(valid_ranges),
                'max': np.max(valid_ranges),
                'timestamp': self.get_clock().now()
            })

            # Keep only recent history
            if len(self.scan_history) > self.max_history:
                self.scan_history = self.scan_history[-self.max_history:]

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()
    rclpy.spin(validator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Sensor Simulation

1. **Match Real Sensor Specifications**: Configure simulation parameters to match real sensor capabilities
2. **Add Realistic Noise**: Include appropriate noise models to make data more realistic
3. **Validate Against Real Data**: Compare simulation output with real sensor data
4. **Optimize Update Rates**: Balance realism with simulation performance
5. **Use Proper Coordinate Frames**: Ensure consistent TF frames across all sensors
6. **Consider Computational Load**: Balance sensor complexity with simulation performance

### Physical Grounding and Simulation-to-Real Mapping

When simulating sensors, consider the mapping to real hardware:

- **Field of View**: Match simulated cameras to real camera specifications
- **Range and Resolution**: Configure sensors to match real hardware capabilities
- **Noise Characteristics**: Include realistic noise models based on real sensor data
- **Update Rates**: Set appropriate update rates that match real sensors
- **Mounting Positions**: Place simulated sensors at the same locations as real sensors

### Troubleshooting Sensor Issues

Common sensor simulation problems and solutions:

- **No Data**: Check topic names and ensure plugins are loaded correctly
- **Wrong Data Type**: Verify message types match expected ROS message types
- **Incorrect TF Frames**: Ensure coordinate frames are properly defined
- **Performance Issues**: Reduce update rates or simplify sensor models
- **Noise Issues**: Adjust noise parameters to match real sensor characteristics

### Summary

This chapter covered sensor simulation in Gazebo and integration with ROS 2, which is essential for creating realistic digital twins of robotic systems. You learned how to configure various types of sensors, process sensor data in ROS 2 nodes, and implement sensor fusion techniques. Proper sensor simulation is crucial for developing and testing perception and navigation algorithms before deployment on real hardware. In the next chapter, we'll explore Unity 3D simulation environments for robotics applications.