# Chapter 1.2: Nodes and Topics

## Understanding Nodes

Nodes are the fundamental building blocks of ROS 2. Each node performs a specific task and communicates with other nodes.

### Creating a Node

Basic node structure:
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

## Topics - Publisher/Subscriber Pattern

Topics enable one-way communication between nodes through a publish/subscribe model.

### Publisher Node

```python
import rclpy
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
```

### Subscriber Node

```python
import rclpy
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

## Key Points

- Nodes run independently
- Topics enable asynchronous communication
- Multiple nodes can publish/subscribe to the same topic
- Message types ensure data compatibility