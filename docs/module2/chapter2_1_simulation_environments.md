# Module 2: Digital Twins – Gazebo & Unity

## Chapter 2.1: Introduction to Simulation Environments

This chapter introduces simulation environments as digital twins for robotics development, focusing on Gazebo Garden as a physics-based simulation platform. Simulation environments are essential for testing and validating robotic systems before deployment on real hardware.

### The Role of Simulation in Robotics

Simulation environments serve as digital twins of the real world, allowing developers to test robotic algorithms, validate system behavior, and train AI models in a safe, controlled, and cost-effective environment. The benefits of simulation include:

- **Safety**: Test dangerous scenarios without risk to hardware or humans
- **Cost-effectiveness**: Avoid expensive hardware damage during development
- **Repeatability**: Run identical experiments multiple times
- **Speed**: Accelerate development by running simulations faster than real-time
- **Control**: Create controlled environments with known ground truth
- **Scalability**: Test multiple robots simultaneously

### Introduction to Gazebo Garden

Gazebo Garden is the latest version of the Gazebo physics simulator, designed specifically for robotics applications. It provides realistic physics simulation, high-quality rendering, and seamless integration with ROS 2.

#### Key Features of Gazebo Garden

1. **Physics Engine**: Uses modern physics engines for accurate simulation
2. **Rendering**: High-quality graphics rendering for visual simulation
3. **Sensors**: Support for various sensor types (cameras, lidar, IMU, etc.)
4. **ROS 2 Integration**: Native support for ROS 2 communication
5. **Plugin Architecture**: Extensible through plugins for custom functionality
6. **World Editor**: Tools for creating and modifying simulation environments

#### Installation and Setup

To install Gazebo Garden with ROS 2 Humble Hawksbill:

```bash
# Install Gazebo Garden
sudo apt update
sudo apt install ros-humble-gazebo-*

# Source ROS 2 and Gazebo
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
```

### Basic Gazebo Concepts

#### Worlds

A world in Gazebo defines the environment where your robot operates. It includes:
- Terrain and static objects
- Lighting conditions
- Physics parameters
- Weather effects (in advanced versions)

Example world file (`.world`):
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a default light source -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box model -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

#### Models

Models in Gazebo represent physical objects including robots, obstacles, and environment elements. They contain:

- **Links**: Rigid bodies with physical properties
- **Joints**: Connections between links
- **Visual elements**: How the model appears
- **Collision elements**: Physical interaction properties
- **Sensors**: Simulated sensor data

### Launching Gazebo with ROS 2

Gazebo can be launched with ROS 2 integration using launch files:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Launch Gazebo server
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': '-r empty.sdf'  # Launch with empty world
        }.items()
    )

    # Launch Gazebo client (GUI)
    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': '-g'  # Launch GUI only
        }.items()
    )

    return LaunchDescription([
        gzserver_launch,
        gzclient_launch
    ])
```

### Connecting to Gazebo from ROS 2 Nodes

To interact with Gazebo from ROS 2 nodes, you can use the ROS Gazebo bridge:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import rclpy.qos

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Publisher to send commands to Gazebo
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/model/robot/cmd_vel',
            rclpy.qos.qos_profile_sensor_data
        )

        # Timer to send commands
        self.timer = self.create_timer(0.1, self.send_command)

    def send_command(self):
        # Create and send a velocity command
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.2  # Turn at 0.2 rad/s

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Working with Gazebo Services

Gazebo provides various services for controlling the simulation:

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState, GetEntityState
from geometry_msgs.msg import Pose, Twist

class GazeboServiceClient(Node):
    def __init__(self):
        super().__init__('gazebo_service_client')

        # Create clients for Gazebo services
        self.reset_simulation_client = self.create_client(
            Empty, '/world/reset'
        )
        self.set_state_client = self.create_client(
            SetEntityState, '/world/set_entity_state'
        )
        self.get_state_client = self.create_client(
            GetEntityState, '/world/get_entity_state'
        )

        # Wait for services to be available
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting again...')

    def reset_simulation(self):
        # Call reset service
        future = self.reset_simulation_client.call_async(Empty.Request())
        return future

    def set_robot_pose(self, entity_name, x, y, z, roll, pitch, yaw):
        # Create request to set entity state
        req = SetEntityState.Request()
        req.state.name = entity_name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z

        # Convert Euler angles to quaternion
        from math import sin, cos
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)

        req.state.pose.orientation.w = cr * cp * cy + sr * sp * sy
        req.state.pose.orientation.x = sr * cp * cy - cr * sp * sy
        req.state.pose.orientation.y = cr * sp * cy + sr * cp * sy
        req.state.pose.orientation.z = cr * cp * sy - sr * sp * cy

        future = self.set_state_client.call_async(req)
        return future

def main(args=None):
    rclpy.init(args=args)
    client = GazeboServiceClient()

    # Example: Reset simulation
    client.reset_simulation()

    # Example: Set robot pose
    client.set_robot_pose('robot', 1.0, 2.0, 0.0, 0.0, 0.0, 1.57)

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Simulation Parameters and Configuration

Gazebo simulation can be configured with various parameters:

```yaml
# Example Gazebo configuration
gazebo:
  physics:
    type: ode
    max_step_size: 0.001
    real_time_factor: 1.0
    real_time_update_rate: 1000
  rendering:
    engine: ogre
    camera:
      near: 0.1
      far: 100.0
  plugins:
    - libgazebo_ros_factory.so
    - libgazebo_ros_force_system.so
```

### Best Practices for Simulation

1. **Validate Simulation Fidelity**: Ensure your simulation accurately represents real-world physics
2. **Start Simple**: Begin with basic environments before adding complexity
3. **Use Ground Truth**: Take advantage of simulation's access to perfect state information
4. **Test Transitions**: Verify that algorithms work in both simulation and reality
5. **Monitor Performance**: Keep simulation running at acceptable speeds
6. **Document Differences**: Keep track of known differences between simulation and reality

### Physical Grounding and Simulation-to-Real Mapping

When working with simulation environments, it's crucial to maintain awareness of the differences between simulation and reality:

- **Physics Accuracy**: Simulated physics may not perfectly match real-world physics
- **Sensor Noise**: Real sensors have noise and imperfections that may not be modeled
- **Timing**: Simulation timing may differ from real-time execution
- **Environmental Factors**: Real environments have lighting, weather, and other conditions that may not be modeled
- **Hardware Limitations**: Real hardware has computational and power constraints

### Visualization with rviz2 and Gazebo

You can visualize Gazebo simulation data in rviz2:

```bash
# Launch Gazebo
ros2 launch ros_gz_sim gz_sim.launch.py

# In another terminal, launch rviz2
ros2 run rviz2 rviz2

# Add displays for topics published by Gazebo models
```

### Summary

This chapter introduced Gazebo Garden as a physics-based simulation environment for robotics development. You learned about the fundamental concepts of simulation, how to set up Gazebo with ROS 2, and how to interact with the simulation environment. Simulation is a critical tool for robotics development, allowing you to test and validate systems safely before deployment on real hardware. In the next chapter, we'll explore robot modeling and URDF creation, which is essential for representing robots in simulation environments.