# Module 2: Digital Twins – Gazebo & Unity

## Chapter 2.2: Robot Modeling and URDF Creation

This chapter focuses on creating robot models using URDF (Unified Robot Description Format), which is essential for representing robots in simulation environments like Gazebo. URDF defines the physical and visual properties of a robot, enabling accurate simulation and visualization.

### Understanding URDF

URDF (Unified Robot Description Format) is an XML-based format for representing robot models in ROS. It defines:
- Physical structure (links and joints)
- Visual appearance
- Collision properties
- Inertial properties
- Sensor mounting points
- Actuator specifications

### URDF Structure

A basic URDF file consists of links connected by joints:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link - this is the reference frame -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- A second link connected by a joint -->
  <link name="sensor_mount">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- Joint connecting the links -->
  <joint name="sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_mount"/>
    <origin xyz="0.2 0.0 0.1" rpy="0 0 0"/>
  </joint>
</robot>
```

### Link Elements

Links represent rigid bodies in the robot model. Each link can have multiple sub-elements:

#### Visual Element
Defines how the link appears in visualization tools:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Can be box, cylinder, sphere, or mesh -->
    <box size="0.5 0.5 0.2"/>
  </geometry>
  <material name="green">
    <color rgba="0 1 0 1"/>
  </material>
</visual>
```

#### Collision Element
Defines the collision properties for physics simulation:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.5 0.5 0.2"/>
  </geometry>
</collision>
```

#### Inertial Element
Defines the mass and inertial properties for physics simulation:

```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

### Joint Elements

Joints connect links and define their relative motion. Common joint types include:

#### Fixed Joint
A joint with no degrees of freedom:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="base_link"/>
  <child link="fixed_part"/>
  <origin xyz="0.1 0.0 0.0" rpy="0 0 0"/>
</joint>
```

#### Revolute Joint
A joint with one rotational degree of freedom:

```xml
<joint name="revolute_joint" type="revolute">
  <parent link="base_link"/>
  <child link="rotating_part"/>
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
</joint>
```

#### Continuous Joint
Like revolute but without limits:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base_link"/>
  <child link="rotating_part"/>
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>
```

#### Prismatic Joint
A joint with one translational degree of freedom:

```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="sliding_part"/>
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0.0" upper="0.1" effort="10.0" velocity="1.0"/>
</joint>
```

### Creating a Complete Robot Model

Here's an example of a more complex robot model with multiple joints and sensors:

```xml
<?xml version="1.0"?>
<robot name="mobile_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- Wheel joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.1 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.1 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Sensor mount -->
  <link name="laser_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Sensor joint -->
  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0.1 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.02 0.05 0.02"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.05 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

### URDF with Gazebo-Specific Elements

When using URDF with Gazebo, you can add Gazebo-specific elements:

```xml
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo reference="left_wheel">
  <material>Gazebo/Black</material>
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- Gazebo plugin for differential drive -->
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>robot</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
      <remapping>odom:=odom</remapping>
    </ros>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
  </plugin>
</gazebo>

<!-- Gazebo plugin for laser scanner -->
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>robot</namespace>
        <remapping>scan:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### Creating URDF Files with xacro

For complex robots, xacro (XML Macros) can simplify URDF creation:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">
  <!-- Define properties -->
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_length" value="0.5"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.05"/>
  <xacro:property name="wheel_width" value="0.02"/>

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix *origin">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="origin"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="left">
    <origin xyz="0 ${base_length/2} 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="left_wheel"/>
  </xacro:wheel>

  <xacro:wheel prefix="right">
    <origin xyz="0 -${base_length/2} 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="right_wheel"/>
  </xacro:wheel>
</robot>
```

### Validating URDF Files

You can validate your URDF files using ROS tools:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Display the URDF in a tree structure
urdf_to_graphiz /path/to/robot.urdf

# Visualize the robot model
ros2 run rviz2 rviz2
```

### Loading URDF in ROS 2

To use your URDF in ROS 2, you typically load it into a parameter server:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Load robot description from parameter
        self.declare_parameter('robot_description', '')
        self.robot_description = self.get_parameter('robot_description').value

        # Create publishers and broadcasters
        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)
        self.broadcaster = TransformBroadcaster(self)

        # Timer to publish joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = ['left_wheel_joint', 'right_wheel_joint']
        msg.position = [0.0, 0.0]  # Placeholder values
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()

    # Set robot description parameter
    with open('/path/to/robot.urdf', 'r') as file:
        robot_desc = file.read()
    node.set_parameters([rclpy.parameter.Parameter('robot_description', value=robot_desc)])

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for URDF Creation

1. **Use Meaningful Names**: Use descriptive names for links and joints
2. **Start Simple**: Begin with a basic model and add complexity gradually
3. **Verify Inertial Properties**: Ensure mass and inertia values are realistic
4. **Test in Simulation**: Validate your model in Gazebo before complex use
5. **Use xacro for Complex Models**: Simplify complex models with macros
6. **Validate Regularly**: Check your URDF frequently during development
7. **Document Your Model**: Include comments explaining the purpose of different parts

### Physical Grounding and Simulation-to-Real Mapping

When creating URDF models, consider how they'll map to real hardware:

- **Accurate Dimensions**: Ensure model dimensions match real hardware
- **Realistic Inertial Properties**: Use measured or calculated mass/inertia values
- **Correct Joint Limits**: Set limits that match real hardware capabilities
- **Sensor Placement**: Position simulated sensors at the same locations as real sensors
- **Material Properties**: Use realistic friction and collision properties

### Troubleshooting Common URDF Issues

- **Missing Joints**: Ensure all links are connected in a tree structure
- **Inertial Issues**: Verify that all links have proper inertial elements
- **Collision Issues**: Make sure collision elements are properly defined
- **Visualization Problems**: Check that visual elements have proper geometry and materials

### Summary

This chapter covered the fundamentals of creating robot models using URDF, which is essential for representing robots in simulation environments. You learned about the structure of URDF files, different joint types, how to add Gazebo-specific elements, and how to use xacro for complex models. Proper robot modeling is crucial for accurate simulation and forms the foundation for effective robotics development. In the next chapter, we'll explore physics simulation and environment modeling in more detail.