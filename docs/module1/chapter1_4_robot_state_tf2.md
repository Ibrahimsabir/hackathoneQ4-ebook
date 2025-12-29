# Chapter 1.4: Robot State and TF2

## Robot State Publisher

The robot state publisher broadcasts information about robot joints and links.

### Basic Robot State Publisher

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to joint states
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)

        # Timer for broadcasting transforms
        self.timer = self.create_timer(0.1, self.broadcast_transforms)

    def joint_state_callback(self, msg):
        # Process joint state messages
        self.joint_positions = dict(zip(msg.name, msg.position))

    def broadcast_transforms(self):
        # Create and broadcast transforms
        t = TransformStamped()

        # Define transform properties
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'link1'

        # Set translation and rotation
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Quaternion (x, y, z, w)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Send transform
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()
```

## TF2 - Transform Framework

TF2 manages coordinate frames and transforms between them.

### Transform Listener

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PointStamped

class TransformListenerNode(Node):
    def __init__(self):
        super().__init__('transform_listener')

        # Initialize TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically lookup transforms
        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        try:
            # Lookup transform from base_link to laser_frame
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                'laser_frame',
                rclpy.time.Time())

            self.get_logger().info(f'Translation: {trans.transform.translation}')
            self.get_logger().info(f'Rotation: {trans.transform.rotation}')

        except Exception as e:
            self.get_logger().error(f'Could not lookup transform: {str(e)}')
```

## Key Concepts

- **Frames**: Coordinate systems attached to robot parts
- **Transforms**: Relationships between frames
- **Static transforms**: Fixed relationships between frames
- **Dynamic transforms**: Changing relationships (like joint movements)