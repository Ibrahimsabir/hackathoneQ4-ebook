# Module 2: Digital Twins – Gazebo & Unity

## Chapter 2.3: Physics Simulation and Environment Modeling

This chapter explores the physics simulation capabilities of Gazebo Garden and how to create realistic simulation environments for robotics applications. Physics simulation is crucial for accurately modeling robot interactions with the environment.

### Understanding Physics Simulation in Gazebo

Physics simulation in Gazebo allows you to model the behavior of objects in a virtual environment, including:
- Collision detection and response
- Gravitational effects
- Friction and contact forces
- Joint constraints and dynamics
- Sensor simulation with realistic noise models

### Physics Engine Configuration

Gazebo supports multiple physics engines. The most common are:
- **ODE (Open Dynamics Engine)**: Default, good for most applications
- **Bullet**: Good for complex contact scenarios
- **DART**: Advanced physics simulation
- **Simbody**: For biomechanics applications

You can configure physics parameters in your world file:

```xml
<sdf version="1.7">
  <world name="physics_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun light -->
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Collision Properties and Parameters

Collision properties determine how objects interact when they come into contact. Key parameters include:

#### Friction Properties
```xml
<gazebo reference="link_name">
  <mu1>0.5</mu1>  <!-- Primary friction coefficient -->
  <mu2>0.5</mu2>  <!-- Secondary friction coefficient -->
  <fdir1>1 0 0</fdir1>  <!-- Friction direction -->
</gazebo>
```

#### Contact Properties
```xml
<gazebo reference="link_name">
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>      <!-- Contact damping -->
  <max_vel>10.0</max_vel>        <!-- Maximum contact correction velocity -->
  <min_depth>0.001</min_depth>   <!-- Minimum contact depth -->
</gazebo>
```

### Creating Physics-Enabled Models

To make your models behave realistically in simulation, you need to define proper inertial properties:

```xml
<link name="physical_link">
  <!-- Visual properties -->
  <visual name="visual">
    <geometry>
      <mesh filename="package://my_robot_description/meshes/robot_part.dae"/>
    </geometry>
  </visual>

  <!-- Collision properties -->
  <collision name="collision">
    <geometry>
      <mesh filename="package://my_robot_description/meshes/robot_part_collision.dae"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="1.5"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia
      ixx="0.01" ixy="0.0" ixz="0.0"
      iyy="0.01" iyz="0.0"
      izz="0.02"/>
  </inertial>
</link>
```

### Advanced Physics Configuration

For more complex scenarios, you can configure advanced physics properties:

```xml
<gazebo reference="robot_base">
  <!-- Bounce properties -->
  <bounce>
    <restitution_coefficient>0.2</restitution_coefficient>
    <threshold>1.0</threshold>
  </bounce>

  <!-- ODE specific parameters -->
  <ode_kinematic>false</ode_kinematic>
  <max_contacts>10</max_contacts>
</gazebo>
```

### Environment Modeling

Creating realistic environments involves modeling various physical elements:

#### Terrain and Ground Models
```xml
<model name="uneven_terrain">
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>10 10 1</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>10 10 1</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

#### Obstacles and Barriers
```xml
<model name="obstacle">
  <pose>2 3 0 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.5 0.5 1.0</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.5 0.5 1.0</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
        <diffuse>0.8 0.2 0.2 1</diffuse>
        <specular>0.8 0.2 0.2 1</specular>
      </material>
    </visual>
    <inertial>
      <mass value="5.0"/>
      <inertia
        ixx="0.5" ixy="0.0" ixz="0.0"
        iyy="0.5" iyz="0.0"
        izz="0.5"/>
    </inertial>
  </link>
</model>
```

### Sensor Simulation with Physics

Sensors in Gazebo interact with the physics simulation to provide realistic data:

#### Camera Sensor
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

#### Lidar Sensor
```xml
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
        <max>30.0</max>
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

### Physics Simulation Best Practices

#### 1. Time Step Considerations
```xml
<physics type="ode">
  <!-- Smaller time steps provide more accurate simulation but require more computation -->
  <max_step_size>0.001</max_step_size>  <!-- 1ms time step -->
  <real_time_factor>1.0</real_time_factor>  <!-- Run at real-time speed -->
</physics>
```

#### 2. Stability and Performance Balance
```xml
<physics type="ode">
  <ode>
    <solver>
      <!-- Increase iterations for more accurate physics but slower simulation -->
      <iters>20</iters>
      <!-- Lower SOR (Successive Over-Relaxation) for more stability -->
      <sor>1.2</sor>
    </solver>
    <constraints>
      <!-- ERP (Error Reduction Parameter) - higher values correct errors faster -->
      <erp>0.2</erp>
      <!-- CFM (Constraint Force Mixing) - higher values make constraints softer -->
      <cfm>1e-6</cfm>
    </constraints>
  </ode>
</physics>
```

### Advanced Physics Simulation Techniques

#### Joint Dynamics
```xml
<joint name="motorized_joint" type="revolute">
  <parent link="base_link"/>
  <child link="arm_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>

<!-- Gazebo plugin for joint control -->
<gazebo>
  <plugin name="joint_control" filename="libgazebo_ros_joint_position.so">
    <command_topic>joint_position/command</command_topic>
    <state_topic>joint_position/state</state_topic>
    <joint_name>motorized_joint</joint_name>
  </plugin>
</gazebo>
```

#### Custom Physics Plugins
```xml
<gazebo>
  <plugin name="custom_physics_plugin" filename="libcustom_physics_plugin.so">
    <robot_namespace>/my_robot</robot_namespace>
    <parameter_name>value</parameter_name>
  </plugin>
</gazebo>
```

### Environment Interaction Modeling

Modeling how robots interact with their environment:

#### Grasping and Manipulation
```xml
<gazebo reference="gripper_finger">
  <mu1>1.0</mu1>
  <mu2>1.0</mu2>
  <fdir1>1 0 0</fdir1>
  <max_vel>100.0</max_vel>
  <min_depth>0.001</min_depth>
</gazebo>
```

#### Terrain Interaction
```xml
<gazebo reference="wheel_link">
  <!-- High friction for good traction -->
  <mu1>1.0</mu1>
  <mu2>1.0</mu2>
  <!-- Contact properties for realistic wheel-ground interaction -->
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>
```

### Physics Validation and Tuning

Validating your physics simulation against real-world behavior:

```python
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Twist
import time

class PhysicsValidator(Node):
    def __init__(self):
        super().__init__('physics_validator')

        # Client to get entity state from Gazebo
        self.get_state_client = self.create_client(
            GetEntityState, '/world/get_entity_state'
        )

        # Publisher to send commands
        self.cmd_pub = self.create_publisher(
            Twist, '/robot/cmd_vel', 10
        )

        # Timer to run validation
        self.timer = self.create_timer(0.1, self.validate_physics)

    def validate_physics(self):
        # Get robot state from Gazebo
        req = GetEntityState.Request()
        req.name = 'robot'
        req.reference_frame = 'world'

        future = self.get_state_client.call_async(req)
        # Process results to validate physics behavior
        # Compare with expected values based on applied forces

def main(args=None):
    rclpy.init(args=args)
    validator = PhysicsValidator()
    rclpy.spin(validator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Optimization

Optimizing physics simulation for better performance:

1. **Reduce Complexity**: Use simpler collision geometries where possible
2. **Adjust Time Steps**: Balance accuracy with performance
3. **Limit Update Rates**: Don't update sensors faster than necessary
4. **Use Appropriate Solvers**: Choose physics parameters based on simulation needs

### Physical Grounding and Simulation-to-Real Mapping

When configuring physics simulation, consider the mapping to real-world behavior:

- **Material Properties**: Set friction and contact parameters to match real materials
- **Inertial Properties**: Use measured or calculated mass/inertia values
- **Sensor Noise**: Add realistic noise models that match real sensors
- **Timing**: Consider real-time constraints when designing simulation parameters
- **Environmental Conditions**: Account for real-world factors like uneven terrain

### Troubleshooting Physics Issues

Common physics simulation problems and solutions:

- **Objects Falling Through Ground**: Check collision geometry and contact parameters
- **Unstable Simulation**: Reduce time step or adjust solver parameters
- **Objects Floating**: Verify inertial properties and gravity settings
- **Joints Behaving Incorrectly**: Check joint limits and dynamics parameters

### Summary

This chapter covered the physics simulation capabilities of Gazebo Garden and how to create realistic simulation environments. You learned about configuring physics engines, setting collision properties, modeling environments, and optimizing simulation performance. Physics simulation is fundamental to creating accurate digital twins for robotics development, allowing you to test and validate systems before deployment on real hardware. In the next chapter, we'll explore sensor simulation and integration with ROS 2.