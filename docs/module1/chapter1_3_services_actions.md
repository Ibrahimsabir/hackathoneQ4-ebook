# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.3: Services, Actions, and Parameters

This chapter explores two important communication patterns in ROS 2: services for synchronous request/response communication and actions for long-running tasks with feedback. We'll also cover parameters, which provide a way to configure nodes at runtime.

### Services: Request/Response Communication

Services in ROS 2 implement a request/response communication pattern, which is synchronous. When a client sends a request to a service, it waits for a response before continuing. This is different from topics, which are asynchronous and continuous.

#### Service Architecture

A service consists of:
- **Service Server**: A node that provides the service functionality
- **Service Client**: A node that calls the service
- **Service Interface**: Defines the request and response message types

The service interface is defined in a `.srv` file with the following structure:
```
# Request message
string request_field
---
# Response message
int32 response_field
```

#### Creating a Service Server

Here's an example of a simple service server that adds two integers:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Creating a Service Client

And here's the corresponding client that calls the service:

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Actions: Long-Running Tasks with Feedback

Actions are designed for long-running tasks that provide feedback during execution and return a result when completed. They're ideal for tasks like navigation, where you want to track progress and potentially cancel the task.

#### Action Architecture

An action has three parts:
- **Goal**: The request to start the action
- **Feedback**: Messages sent during execution
- **Result**: The final outcome of the action

The action interface is defined in a `.action` file:
```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] sequence
```

#### Creating an Action Server

Here's an example of an action server that generates a Fibonacci sequence:

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info('Publishing feedback: {0}'.format(feedback_msg.sequence))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))

        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    fibonacci_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Creating an Action Client

And the corresponding action client:

```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.sequence))

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

### Parameters: Runtime Configuration

Parameters in ROS 2 allow you to configure nodes at runtime without recompiling. They can be set at launch time, changed dynamically, and accessed programmatically.

#### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'turtlebot4')
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('safety_distance', 0.3)

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety distance: {self.safety_distance}')

        # Set up parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.INTEGER:
                if param.value > 1.0:
                    return SetParametersResult(successful=False, reason='Max velocity too high')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    param_node = ParameterNode()
    rclpy.spin(param_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### When to Use Each Communication Pattern

- **Topics**: Use for continuous data streams like sensor data, robot state, or other ongoing information
- **Services**: Use for tasks that have a clear request/response pattern and complete relatively quickly
- **Actions**: Use for long-running tasks that need feedback, can be canceled, or have intermediate results
- **Parameters**: Use for configuration that might need to be changed at runtime

### Physical Grounding and Simulation-to-Real Mapping

When working with services, actions, and parameters, consider how they'll behave in both simulation and real hardware:

- Services on real hardware might take longer to respond due to processing time
- Actions might have different execution times in simulation vs. real hardware
- Parameters might need different values when transitioning from simulation to real hardware
- Error handling becomes more critical with real hardware, where components might fail

### Summary

This chapter covered the three main communication patterns in ROS 2 beyond topics: services for synchronous request/response communication, actions for long-running tasks with feedback, and parameters for runtime configuration. These communication patterns provide the tools needed to build complex robotic systems with well-defined interfaces between components. In the next chapter, we'll explore robot state publishing and coordinate transformations.