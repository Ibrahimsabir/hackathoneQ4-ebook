# Module 1: ROS 2 – The Robotic Nervous System

## Chapter 1.1: ROS 2 Architecture Basics

ROS 2 (Robot Operating System 2) is the communication framework for robotic systems. It serves as the "nervous system" connecting different robot components.

### Core Concepts

#### Nodes
Nodes are processes that perform specific computations. Each node handles a dedicated task and communicates with other nodes.

Common node types:
- Sensor processing
- Motor control
- Path planning
- Perception

#### Packages
Packages contain libraries, executables, and configuration files for specific functionality.

#### Workspaces
Workspaces are directories where you develop and build packages.

### Communication Patterns

ROS 2 uses three main communication methods:

1. **Topics**: Continuous data streams (like sensor data)
2. **Services**: Request/response communication
3. **Actions**: Long-running tasks with feedback

### Setup

Install ROS 2 Humble Hawksbill with Python 3.10+ and source the setup script:

```bash
source /opt/ros/humble/setup.bash
```

### Key Benefits

- Distributed architecture
- Modularity
- Fault tolerance
- Scalability

This foundation enables complex robotic systems to communicate effectively.