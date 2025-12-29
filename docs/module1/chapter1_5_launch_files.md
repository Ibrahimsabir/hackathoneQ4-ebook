# Chapter 1.5: Launch Files

## Launch Files Overview

Launch files allow you to start multiple nodes with a single command.

### Python Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),

        # Launch robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]),

        # Launch joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher'),
    ])
```

### YAML Launch File

```yaml
launch:
  - node:
      pkg: "robot_state_publisher"
      exec: "robot_state_publisher"
      name: "robot_state_publisher"
      parameters:
        - "config/robot_params.yaml"

  - node:
      pkg: "joint_state_publisher"
      exec: "joint_state_publisher"
      name: "joint_state_publisher"
```

## Key Concepts

- **Launch Arguments**: Configurable parameters for launch files
- **Node Groups**: Organize related nodes
- **Conditions**: Conditional launching based on parameters
- **Event Handlers**: Respond to node events