# Chapter 1.6: Testing and Debugging

## Testing ROS 2 Nodes

Testing is critical for ensuring robust robotic systems.

### Unit Testing

```python
import unittest
import rclpy
from rclpy.node import Node
from your_package.your_node import YourNode

class TestYourNode(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = YourNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        self.assertEqual(self.node.get_name(), 'your_node_name')

    def test_publish_subscribe(self):
        # Test that messages are published and subscribed correctly
        # Implementation depends on your specific node
        pass

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from your_package.test_nodes import PublisherNode, SubscriberNode

class TestNodeIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.publisher_node = PublisherNode()
        self.subscriber_node = SubscriberNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.publisher_node)
        self.executor.add_node(self.subscriber_node)

    def tearDown(self):
        self.publisher_node.destroy_node()
        self.subscriber_node.destroy_node()
        rclpy.shutdown()

    def test_end_to_end(self):
        # Test full publisher-subscriber flow
        self.publisher_node.publish_test_message()
        self.executor.spin_once(timeout_sec=1.0)
        self.assertTrue(self.subscriber_node.received_message)
```

## Debugging Techniques

### Logging

```python
import rclpy
from rclpy.node import Node

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        # Different log levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')
```

### Using ros2doctor

Check system health:
```bash
ros2 doctor
```

### Topic Inspection

Monitor topics:
```bash
# Echo messages
ros2 topic echo /topic_name

# Check topic info
ros2 topic info /topic_name

# List all topics
ros2 topic list
```

## Common Debugging Tools

- **RViz2**: Visualize robot state and sensor data
- **rqt**: GUI tools for monitoring and debugging
- **ros2 bag**: Record and replay data
- **ros2 lifecycle**: Manage node lifecycles
- **ros2 param**: View and modify parameters

## Best Practices

- Test individual components before system integration
- Use mock objects for dependencies
- Log appropriately for different severity levels
- Monitor system resources during testing
- Use linters and code analyzers