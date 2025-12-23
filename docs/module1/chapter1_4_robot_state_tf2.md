# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.4: Robot State Publishing and tf2

This chapter covers the essential concepts of robot state publishing and coordinate transformations using tf2 (Transform Library 2), which is critical for understanding how robots perceive and navigate their environment.

### Understanding Robot State

Robot state refers to the complete description of a robot's configuration and position in space. This includes:
- Joint positions and velocities
- Robot pose (position and orientation) in various coordinate frames
- Sensor positions and orientations
- Any other relevant kinematic information

### The Robot State Publisher

The Robot State Publisher (robot_state_publisher) is a ROS 2 node that takes the robot's joint positions and uses the URDF (Unified Robot Description Format) model to compute the forward kinematics of the robot. It then publishes the resulting transforms to tf2.

#### Joint State Message

The robot_state_publisher typically receives joint states through the `sensor_msgs/JointState` message:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer to publish joint states at regular intervals
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        # Initialize joint names
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_angles = [0.0, 0.0, 0.0]  # Initial joint angles

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_angles
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint angles for demonstration (e.g., oscillating motion)
        time_sec = self.get_clock().now().nanoseconds / 1e9
        self.joint_angles[0] = math.sin(time_sec) * 0.5
        self.joint_angles[1] = math.cos(time_sec) * 0.3
        self.joint_angles[2] = math.sin(time_sec * 0.5) * 0.2

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Understanding tf2 (Transform Library 2)

tf2 is the second-generation Transform Library in ROS 2, which allows users to keep track of multiple coordinate frames over time. tf2 maintains the relationship between coordinate frames in a tree structure buffered in time, and lets the user transform points, vectors, etc. between any two coordinate frames at any desired point in time.

#### Key Concepts in tf2

1. **Coordinate Frames**: Named reference frames that define positions and orientations
2. **Transforms**: Relationships between frames (position and orientation)
3. **Static Transforms**: Relationships that don't change over time
4. **Dynamic Transforms**: Relationships that change over time

#### Publishing Transforms

Here's an example of publishing transforms using tf2:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to publish transforms
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.broadcast_transform)

    def broadcast_transform(self):
        # Create transform from base_link to laser_frame
        t = TransformStamped()

        # Set header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'laser_frame'

        # Set transform (position and orientation)
        t.transform.translation.x = 0.1  # 10cm forward from base
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2  # 20cm up from base

        # Simple rotation (no rotation in this example)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Send the transform
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = FramePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscribing to and Using Transforms

To use transforms in your nodes, you need to create a TransformListener:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # Import to use tf2 with geometry_msgs

class TransformDemo(Node):
    def __init__(self):
        super().__init__('transform_demo')

        # Create tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to perform transform
        timer_period = 1.0  # 1Hz
        self.timer = self.create_timer(timer_period, self.transform_point)

    def transform_point(self):
        try:
            # Define a point in the laser_frame
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = 'laser_frame'
            point_stamped.point.x = 1.0
            point_stamped.point.y = 0.0
            point_stamped.point.z = 0.0

            # Transform the point to base_link frame
            point_out = self.tf_buffer.transform(
                point_stamped,
                'base_link',
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            self.get_logger().info(
                f'Point in base_link: ({point_out.point.x:.2f}, '
                f'{point_out.point.y:.2f}, {point_out.point.z:.2f})'
            )

        except Exception as e:
            self.get_logger().info(f'Could not transform: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TransformDemo()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Static Transform Publisher

For transforms that don't change over time (like the position of a sensor on a robot), you can use a static transform publisher:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

class StaticFramePublisher(Node):
    def __init__(self):
        super().__init__('static_frame_publisher')

        # Create static transform broadcaster
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Publish the static transform
        self.publish_static_transform()

    def publish_static_transform(self):
        # Create transform
        t = TransformStamped()

        # Set header (timestamp is typically 0 for static transforms)
        t.header.stamp = rclpy.time.Time().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_frame'

        # Set transform (position and orientation)
        t.transform.translation.x = 0.1  # 10cm forward from base
        t.transform.translation.y = 0.05  # 5cm to the left
        t.transform.translation.z = 0.2  # 20cm up

        # Set rotation (90 degrees around z-axis)
        import math
        from tf2_ros import transform_to_quaternion
        from geometry_msgs.msg import Quaternion

        # Convert roll, pitch, yaw to quaternion
        # For 90 degree rotation around z-axis
        roll, pitch, yaw = 0.0, 0.0, math.pi/2
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy

        # Send the static transform
        self.tf_static_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = StaticFramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### URDF and Robot Description

URDF (Unified Robot Description Format) is an XML format for representing a robot model. The robot_state_publisher uses URDF to automatically compute and publish the transforms between robot links based on joint positions.

A simple URDF example:
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Link connected via a joint -->
  <link name="sensor_mount">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting the links -->
  <joint name="sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_mount"/>
    <origin xyz="0.2 0.0 0.1" rpy="0 0 0"/>
  </joint>
</robot>
```

### Practical Example: Complete Robot State Publisher

Here's a more complete example that combines joint state publishing and transform publishing:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class CompleteRobotStatePublisher(Node):
    def __init__(self):
        super().__init__('complete_robot_state_publisher')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.broadcaster = TransformBroadcaster(self)

        # Timer to publish state
        timer_period = 0.05  # 20Hz
        self.timer = self.create_timer(timer_period, self.publish_robot_state)

        # Robot parameters
        self.wheel_radius = 0.05
        self.wheel_separation = 0.3  # Distance between wheels
        self.encoder_resolution = 4096  # Counts per revolution

    def publish_robot_state(self):
        # Get current time
        time = self.get_clock().now()

        # Create joint state message
        msg = JointState()
        msg.name = ['left_wheel_joint', 'right_wheel_joint', 'caster_wheel_joint']
        msg.position = [0.0, 0.0, 0.0]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0]
        msg.header.stamp = time.to_msg()
        msg.header.frame_id = 'base_link'

        # Simulate some joint movement
        current_time = time.nanoseconds / 1e9
        msg.position[0] = math.sin(current_time)  # Left wheel
        msg.position[1] = math.cos(current_time)  # Right wheel
        msg.position[2] = math.sin(current_time * 0.5)  # Caster wheel

        # Publish joint state
        self.joint_pub.publish(msg)

        # Publish transforms
        self.publish_transforms(time)

    def publish_transforms(self, time):
        # Base link to laser frame transform
        t = TransformStamped()
        t.header.stamp = time.to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'laser_frame'
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = CompleteRobotStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Physical Grounding and Simulation-to-Real Mapping

When working with robot state and tf2, it's crucial to maintain proper mapping between simulation and real hardware:

- Joint limits and ranges must match between simulation and reality
- Transform accuracy is critical for proper sensor fusion and navigation
- Timing differences between simulation and real hardware can affect transform accuracy
- Real sensors have mounting offsets that must be accurately represented in transforms
- Calibration procedures may be needed to fine-tune transform parameters

### Visualization with rviz2

In rvdf2, you can visualize the robot's tf tree to verify that transforms are being published correctly:

```bash
ros2 run rviz2 rviz2
```

In rviz2, add a TF display and set the fixed frame to visualize the transform tree. This is invaluable for debugging transform issues.

### Summary

This chapter covered robot state publishing and coordinate transformations using tf2, which are essential for understanding how robots perceive their environment and navigate. You learned how to publish joint states, broadcast transforms, and work with the tf2 library to transform data between coordinate frames. These concepts are fundamental to robotics and will be essential as you develop more complex robotic systems in subsequent modules.