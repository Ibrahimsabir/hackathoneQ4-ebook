# Chapter 2.1: Simulation Environments

## Introduction to Simulation

Simulation environments are crucial for developing and testing robotic systems before deployment on real hardware.

### Popular Simulation Platforms

- **Gazebo**: Physics-based simulation with realistic dynamics
- **Unity**: Game engine-based simulation with advanced graphics
- **Webots**: General-purpose robot simulator with built-in physics

### Gazebo Setup

```bash
# Install Gazebo
sudo apt-get install gazebo11 libgazebo11-dev

# Launch Gazebo
gazebo
```

### Basic Gazebo World

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="simple_world">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
      </link>
    </model>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>
  </world>
</sdf>
```

## Key Concepts

- **Physics Engines**: ODE, Bullet, SimBody
- **Sensors**: Cameras, LiDAR, IMUs
- **Actuators**: Motors, servos
- **Models**: Robots, objects, environments