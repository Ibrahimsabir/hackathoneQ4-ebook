# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.2: ROS 2 Nodes, Topics, and Message Passing

This chapter focuses on creating and managing ROS 2 nodes and implementing the publish/subscribe communication pattern, which is fundamental to ROS 2's architecture.

### Understanding ROS 2 Nodes

A node in ROS 2 is an executable that uses the ROS 2 client library (like rclpy for Python). Nodes are the fundamental building blocks of a ROS 2 application. Each node typically performs a specific task and communicates with other nodes through topics, services, or actions.

### Creating Your First ROS 2 Node in Python

To create a ROS 2 node using Python and rclpy, you need to:

1. Import the rclpy client library
2. Initialize the ROS 2 client library
3. Create a node that inherits from `rclpy.Node`
4. Add functionality to your node
5. Spin the node to process callbacks
6. Clean up resources when done

Here's a simple example of a ROS 2 publisher node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber Node

Here's a corresponding subscriber node that listens to the publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topic Communication Pattern

Topics in ROS 2 use a publish/subscribe communication pattern. Publishers send data to a topic, and subscribers receive data from a topic. This is a many-to-many relationship - multiple publishers can publish to the same topic, and multiple subscribers can listen to the same topic.

The communication is asynchronous, meaning publishers don't wait for subscribers to receive messages. This makes the system more robust but requires careful consideration of message rates and buffering.

### Quality of Service (QoS) Settings

ROS 2 provides Quality of Service (QoS) settings that allow you to configure how messages are delivered. Important QoS settings include:

- **Reliability**: Whether messages should be reliably delivered (RELIABLE) or best-effort (BEST_EFFORT)
- **Durability**: Whether to keep messages for late-joining subscribers (TRANSIENT_LOCAL) or only send new messages (VOLATILE)
- **History**: How many messages to keep in the queue (KEEP_ALL or KEEP_LAST)
- **Depth**: How many messages to keep in the queue when using KEEP_LAST

Example with QoS settings:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

self.publisher_ = self.create_publisher(String, 'topic', qos_profile)
```

### Message Types and Definition

Messages in ROS 2 are defined in `.msg` files and are the data structures that are exchanged between nodes. Common built-in message types include:

- `std_msgs`: Basic data types like String, Int32, Float64, Bool
- `geometry_msgs`: Geometric primitives like Point, Pose, Twist
- `sensor_msgs`: Sensor data types like Image, LaserScan, JointState
- `nav_msgs`: Navigation-related types like Odometry, Path

You can also define custom message types by creating your own `.msg` files in your package.

### Practical Example: Robot Sensor Data Publisher

Let's create a more practical example of a node that simulates publishing sensor data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import random

class RobotSensorPublisher(Node):
    def __init__(self):
        super().__init__('robot_sensor_publisher')

        # Publisher for laser scan data
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)

        # Publisher for robot velocity
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer to publish data at regular intervals
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

        # Initialize scan parameters
        self.scan_msg = LaserScan()
        self.scan_msg.header.frame_id = 'laser_frame'
        self.scan_msg.angle_min = -1.57  # -90 degrees
        self.scan_msg.angle_max = 1.57   # 90 degrees
        self.scan_msg.angle_increment = 0.0174  # 1 degree
        self.scan_msg.range_min = 0.1
        self.scan_msg.range_max = 10.0
        self.scan_msg.ranges = [float('inf')] * 181  # 181 points for -90 to +90 degrees

    def publish_sensor_data(self):
        # Update laser scan with simulated data
        self.scan_msg.header.stamp = self.get_clock().now().to_msg()

        # Simulate some obstacles
        for i in range(len(self.scan_msg.ranges)):
            distance = 2.0 + random.uniform(-0.5, 0.5)
            self.scan_msg.ranges[i] = max(0.1, min(10.0, distance))

        self.scan_publisher.publish(self.scan_msg)

        # Publish a simple velocity command
        vel_msg = Twist()
        vel_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        vel_msg.angular.z = 0.1  # Turn slightly
        self.vel_publisher.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = RobotSensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Visualization with rviz2

ROS 2 includes rviz2, a 3D visualization tool that allows you to visualize robot data in real-time. You can visualize laser scans, robot models, camera images, and many other data types. To launch rviz2:

```bash
rviz2
```

In rviz2, you can add displays for different topics to visualize your robot's sensor data and state.

### Physical Grounding and Simulation-to-Real Mapping

When developing nodes that will eventually run on real hardware, it's important to consider how the simulation maps to reality. For example:

- A simulated laser scanner might publish perfect data, but a real scanner will have noise and occasional missed detections
- Simulation timing might be perfect, but real systems have variable latencies
- Simulation might run faster or slower than real-time depending on computational load

Design your nodes to be robust to these differences by using appropriate QoS settings and error handling.

### Summary

This chapter covered the fundamentals of creating ROS 2 nodes and implementing the publish/subscribe communication pattern. You learned how to create publishers and subscribers, configure QoS settings, and work with different message types. These concepts form the foundation of ROS 2 communication and will be essential as you build more complex robotic systems in subsequent chapters.