# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.6: ROS 2 Testing and Debugging

This chapter covers testing and debugging techniques for ROS 2 applications, which are essential for developing reliable robotic systems.

### Testing in ROS 2

Testing is critical for ensuring the reliability and safety of robotic systems. ROS 2 provides several testing frameworks and methodologies to verify the correctness of your nodes and systems.

#### Unit Testing with Python

ROS 2 uses standard Python testing frameworks like `unittest` and `pytest` for unit testing:

```python
import unittest
import rclpy
from rclpy.node import Node
from example_interfaces.msg import String

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.publisher = self.create_publisher(String, 'test_topic', 10)
        self.subscription = self.create_subscription(
            String, 'test_topic', self.callback, 10)
        self.received_message = None

    def callback(self, msg):
        self.received_message = msg.data

class TestNodeMethods(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = TestNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_message_publish_subscribe(self):
        # Test that publishing a message results in the same message being received
        test_msg = String()
        test_msg.data = 'Hello, ROS 2!'

        # Publish the message
        self.node.publisher.publish(test_msg)

        # Spin to process the message
        rclpy.spin_once(self.node, timeout_sec=0.1)

        # Check that the message was received
        self.assertEqual(self.node.received_message, 'Hello, ROS 2!')

def main():
    unittest.main()

if __name__ == '__main__':
    main()
```

#### Integration Testing

Integration testing verifies that multiple nodes work together correctly:

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from example_interfaces.msg import String

class TestNodeA(Node):
    def __init__(self):
        super().__init__('test_node_a')
        self.publisher = self.create_publisher(String, 'integration_topic', 10)

    def publish_message(self, data):
        msg = String()
        msg.data = data
        self.publisher.publish(msg)

class TestNodeB(Node):
    def __init__(self):
        super().__init__('test_node_b')
        self.subscription = self.create_subscription(
            String, 'integration_topic', self.callback, 10)
        self.received_messages = []

    def callback(self, msg):
        self.received_messages.append(msg.data)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node_a = TestNodeA()
        self.node_b = TestNodeB()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node_a)
        self.executor.add_node(self.node_b)

    def tearDown(self):
        self.node_a.destroy_node()
        self.node_b.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()

    def test_node_communication(self):
        # Publish a message from node A
        test_data = "Integration test message"
        self.node_a.publish_message(test_data)

        # Spin to allow communication to occur
        self.executor.spin_once(timeout_sec=0.2)

        # Check that node B received the message
        self.assertIn(test_data, self.node_b.received_messages)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
```

### Debugging Techniques

Effective debugging is essential for identifying and fixing issues in ROS 2 applications.

#### Using ROS 2 Command Line Tools

ROS 2 provides powerful command-line tools for debugging:

```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# List all topics
ros2 topic list

# Echo messages on a topic to see real-time data
ros2 topic echo /topic_name message_type

# Publish messages to a topic for testing
ros2 topic pub /topic_name message_type '{field: value}'

# List all services
ros2 service list

# Call a service for testing
ros2 service call /service_name service_type '{request_field: value}'

# List all actions
ros2 action list

# Send an action goal
ros2 action send_goal /action_name action_type '{goal_field: value}'
```

#### Using rqt Tools

The rqt suite provides graphical debugging tools:

```bash
# General rqt interface
rqt

# Specific tools:
rqt_graph          # Visualize the node graph
rqt_plot          # Plot numerical values over time
rqt_console       # View ROS 2 log messages
rqt_bag           # Record and play back ROS 2 messages
rqt_reconfigure   # Dynamically reconfigure node parameters
rqt_topic         # View topic information
```

### Debugging with Logging

Proper logging is essential for debugging ROS 2 applications:

```python
import rclpy
from rclpy.node import Node

class DebuggingNode(Node):
    def __init__(self):
        super().__init__('debugging_node')

        # Different log levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')

    def some_method(self):
        # Log with context
        value = 42
        self.get_logger().info(f'Processing value: {value}')

        # Log errors with context
        try:
            # Some operation that might fail
            result = 10 / 0
        except ZeroDivisionError as e:
            self.get_logger().error(f'Division error occurred: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DebuggingNode()

    # Set log level at runtime
    node.set_parameters([Parameter('logger.level', value='debug')])

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Debugging with GDB

For more complex debugging scenarios, you can attach GDB to ROS 2 nodes:

```bash
# Run a node with GDB
gdb --args ros2 run package_name executable_name

# Or use valgrind for memory debugging
valgrind --tool=memcheck ros2 run package_name executable_name
```

### Using ROS 2 Bag for Data Recording and Playback

ROS 2 bag allows you to record and playback data for debugging:

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /topic1 /topic2

# Record with custom options
ros2 bag record -o my_recording /topic1 /topic2 --compression-mode file --compression-format zstd

# Play back a recording
ros2 bag play my_recording

# Play back with specific options
ros2 bag play my_recording --rate 0.5  # Play at half speed
```

### Performance Profiling

To debug performance issues, ROS 2 provides profiling tools:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time

class ProfilingNode(Node):
    def __init__(self):
        super().__init__('profiling_node')

        # Create a timer to measure performance
        self.timer = self.create_timer(0.1, self.performance_callback)
        self.iteration_count = 0
        self.start_time = self.get_clock().now()

    def performance_callback(self):
        # Measure execution time
        start = time.time()

        # Do some work
        result = self.process_data()

        # Calculate execution time
        end = time.time()
        execution_time = (end - start) * 1000  # Convert to milliseconds

        self.get_logger().info(f'Iteration {self.iteration_count}: {execution_time:.2f}ms')

        # Check if execution time is within acceptable bounds
        if execution_time > 10:  # More than 10ms
            self.get_logger().warn(f'Slow execution detected: {execution_time:.2f}ms')

        self.iteration_count += 1

    def process_data(self):
        # Simulate some processing
        import math
        return math.sqrt(12345.6789) * self.iteration_count

def main(args=None):
    rclpy.init(args=args)
    node = ProfilingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Debugging Transform Issues

tf2 transforms can be particularly challenging to debug:

```bash
# View the transform tree
ros2 run tf2_tools view_frames

# Echo transforms between frames
ros2 run tf2_ros tf2_echo source_frame target_frame

# Check transform availability
ros2 run tf2_ros tf2_monitor
```

### Memory and Resource Debugging

Monitor resource usage in ROS 2 systems:

```python
import rclpy
from rclpy.node import Node
import psutil  # Requires: pip install psutil

class ResourceMonitorNode(Node):
    def __init__(self):
        super().__init__('resource_monitor')
        self.timer = self.create_timer(1.0, self.monitor_resources)

    def monitor_resources(self):
        # Get process information
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        # Log resource usage
        self.get_logger().info(
            f'Memory: {memory_info.rss / 1024 / 1024:.2f} MB, '
            f'CPU: {cpu_percent}%'
        )

        # Check for potential memory leaks
        if memory_info.rss > 500 * 1024 * 1024:  # 500MB threshold
            self.get_logger().warn('High memory usage detected')

def main(args=None):
    rclpy.init(args=args)
    node = ResourceMonitorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Debugging Best Practices

1. **Use Appropriate Log Levels**: Use debug for detailed information, info for general operational information, warn for potential issues, and error for actual problems.

2. **Log Contextual Information**: Include relevant variables, timestamps, and state information in your logs.

3. **Test Incrementally**: Test individual components before integrating them into larger systems.

4. **Use Simulation**: Test in simulation before deploying to real hardware.

5. **Monitor System Resources**: Keep track of CPU, memory, and network usage.

6. **Validate Messages**: Verify that published messages conform to expected formats and ranges.

7. **Handle Exceptions Gracefully**: Implement proper error handling and recovery mechanisms.

### Testing and Debugging Tools Configuration

Configure your package for testing by adding appropriate entries to your `package.xml`:

```xml
<test_depend>ament_copyright</test_depend>
<test_depend>ament_flake8</test_depend>
<test_depend>ament_pep257</test_depend>
<test_depend>python3-pytest</test_depend>
<test_depend>ros2launch</test_depend>
```

And in your `CMakeLists.txt`:

```cmake
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_node test/test_node.py)
  set(TEST_ENVIRONMENT "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
  ament_add_pytest_test(test_integration test/test_integration.py
    ENVIRONMENT ${TEST_ENVIRONMENT})
endif()
```

### Physical Grounding and Simulation-to-Real Mapping

When testing and debugging, consider the differences between simulation and real hardware:

- Real hardware has timing constraints that simulation may not fully capture
- Sensor noise and environmental factors are often absent in simulation
- Real hardware can fail in ways that simulation cannot model
- Debugging tools may be limited on embedded hardware
- Resource constraints (CPU, memory) may be more severe on real hardware

### Summary

This chapter covered essential testing and debugging techniques for ROS 2 applications. You learned about unit testing, integration testing, debugging with command-line tools, logging strategies, and performance profiling. Testing and debugging are critical for developing reliable robotic systems, and the tools and techniques covered in this chapter will help you identify and resolve issues in your ROS 2 applications. With a solid foundation in ROS 2 architecture, communication patterns, state management, system organization, and testing/debugging, you're well-equipped to develop complex robotic systems in the subsequent modules.