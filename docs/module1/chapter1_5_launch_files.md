# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.5: Launch Files and System Management

This chapter covers the organization and launching of complex ROS 2 systems using launch files, which provide a structured way to start multiple nodes with specific configurations.

### Understanding Launch Files

Launch files in ROS 2 provide a way to start multiple nodes with specific configurations simultaneously. They allow you to:
- Start multiple nodes with a single command
- Set parameters for each node
- Configure remappings between topics
- Launch nodes conditionally based on conditions
- Include other launch files for modular system composition

### Launch File Syntax and Structure

ROS 2 launch files can be written in Python or XML. Python launch files offer more flexibility and are generally preferred for complex systems.

#### Basic Python Launch File

Here's a simple example of a Python launch file:

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
            description='Use simulation clock if true'
        ),

        # Launch a node
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Launch another node
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            remappings=[
                ('/turtle1/cmd_vel', '/cmd_vel')
            ]
        )
    ])
```

### Launch Arguments and Parameters

Launch arguments allow you to pass values to your launch file at runtime:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='robot1',
        description='Name of the robot'
    )

    # Use launch configurations in nodes
    robot_node = Node(
        package='my_robot_package',
        executable='robot_control',
        name=[LaunchConfiguration('robot_name'), '_control'],
        parameters=[
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_name': LaunchConfiguration('robot_name')
            }
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        robot_node
    ])
```

### Launch Conditions

You can conditionally launch nodes based on arguments:

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    declare_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Whether to launch GUI'
    )

    # Conditional node launch
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('use_gui')),
        package='rviz2',
        executable='rviz2',
        name='rviz2'
    )

    return LaunchDescription([
        declare_gui,
        rviz_node
    ])
```

### Including Other Launch Files

You can include other launch files to create modular system compositions:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include another launch file
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'launch',
                'robot_description.launch.py'
            ])
        ])
    )

    # Include simulation launch
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'launch',
                'simulation.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        robot_launch,
        sim_launch
    ])
```

### Advanced Launch Features

#### Setting Environment Variables

```python
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable('RCUTILS_LOGGING_SEVERITY_THRESHOLD', 'INFO'),

        # Node that uses the environment variable
        Node(
            package='my_package',
            executable='my_node',
            name='my_node'
        )
    ])
```

#### Timer Actions

```python
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch first node immediately
        Node(
            package='my_package',
            executable='first_node',
            name='first_node'
        ),

        # Launch second node after 5 seconds
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='my_package',
                    executable='second_node',
                    name='second_node'
                )
            ]
        )
    ])
```

### Complex Launch File Example

Here's a more comprehensive example of a launch file for a complete robotic system:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    # Get launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    use_rviz = LaunchConfiguration('use_rviz')

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description':
                # In practice, this would load the robot description from a URDF file
                '<robot name="my_robot"><link name="base_link"/></robot>'
            }
        ]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Navigation stack (simplified)
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # RViz
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('my_robot_description'),
            'rviz',
            'view_robot.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Launch description
    ld = LaunchDescription()

    # Add arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_use_rviz)

    # Add nodes
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_state_publisher)
    ld.add_action(nav2_bringup_launch)
    ld.add_action(rviz)

    return ld
```

### Launch File Best Practices

1. **Modular Design**: Break large systems into smaller, reusable launch files
2. **Parameter Configuration**: Use launch arguments for configurable parameters
3. **Conditional Launches**: Use conditions to launch nodes based on runtime arguments
4. **Error Handling**: Include proper error handling for node startup failures
5. **Documentation**: Document launch arguments and their purposes

### Launch File Organization

A well-organized ROS 2 package typically has a `launch` directory with launch files:

```
my_robot_package/
├── launch/
│   ├── robot.launch.py
│   ├── simulation.launch.py
│   ├── navigation.launch.py
│   └── sensors.launch.py
├── config/
├── src/
└── CMakeLists.txt
```

### Running Launch Files

To run a launch file:

```bash
# Basic launch
ros2 launch my_package my_launch_file.py

# With arguments
ros2 launch my_package my_launch_file.py use_sim_time:=true robot_name:=turtlebot4

# List available launch arguments
ros2 launch my_package my_launch_file.py --show-args
```

### System Management Commands

Useful commands for managing launched systems:

```bash
# List all active nodes
ros2 node list

# List all topics
ros2 topic list

# Get information about a specific node
ros2 node info <node_name>

# Get information about a specific topic
ros2 topic info <topic_name>

# Echo messages on a topic
ros2 topic echo <topic_name> <message_type>

# Get all active processes
ros2 lifecycle list <node_name>  # for lifecycle nodes
```

### Physical Grounding and Simulation-to-Real Mapping

When designing launch files, consider how they'll work in both simulation and real hardware:

- Use launch arguments to switch between simulation and real hardware configurations
- Include different parameter sets for simulation vs. real hardware
- Design modular launch files that can be combined differently for different deployment scenarios
- Consider timing differences between simulation and real hardware in your launch sequences

### Summary

This chapter covered launch files and system management in ROS 2, which are essential for organizing and launching complex robotic systems. You learned how to create launch files with arguments, conditions, and modular components. Launch files are crucial for managing complex robotic systems with multiple nodes and configurations, and they form the backbone of system deployment in ROS 2. In the next chapter, we'll explore testing and debugging techniques for ROS 2 applications.