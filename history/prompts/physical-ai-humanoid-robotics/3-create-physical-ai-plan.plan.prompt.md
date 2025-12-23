---
id: 3
title: Create Physical AI Plan
stage: plan
date: 2025-12-23
surface: agent
model: Claude Sonnet 4.5
feature: physical-ai-humanoid-robotics
branch: 1-physical-ai-humanoid-robotics
user: user
command: /sp.plan
labels: [plan,physical-ai,robotics,curriculum]
links:
  spec: specs/1-physical-ai-humanoid-robotics/spec.md
  plan: book_plan.md
  ticket: null
  adr: null
  pr: null
files:
 - book_plan.md
 - specs/1-physical-ai-humanoid-robotics/spec.md
 - .specify/memory/constitution.md
tests:
---

## Prompt

```text
You are a senior AI curriculum architect and book project planner.

INPUT:
- constitution.md (strict rules enforced)
- spec.md (Spec-Kit Plus specification for the book)

YOUR OBJECTIVE:
Create a COMPLETE book creation plan that will guide AI in generating
the book module-by-module, chapter-by-chapter, week-by-week.

The plan must include:
1. Modules (exactly as defined in spec and constitution)
2. Chapters for each module
3. Weekly breakdowns for each chapter
4. Learning outcomes for each chapter/week
5. Mandatory tools, platforms, or hardware references for each chapter
6. Capstone integration points
7. Dependencies between chapters and modules
8. Constraints that must be followed (from constitution and spec)

OUTPUT REQUIREMENTS:
- Output ONE file: book_plan.md
- Use Markdown format
- Use headings, subheadings, and bullet points
- Be structured and actionable
- Do not write any chapter content
- Strictly respect the module, week, and constitution rules
- Tasks must be actionable by AI in future steps

MANDATORY SECTIONS:
# Overview
# Module 1: ROS 2 – The Robotic Nervous System
  - Chapters
  - Weekly breakdown
  - Learning outcomes
  - Required tools/hardware
# Module 2: Digital Twins – Gazebo & Unity
  - Chapters
  - Weekly breakdown
  - Learning outcomes
  - Required tools/hardware
# Module 3: AI Robot Brain – NVIDIA Isaac
  - Chapters
  - Weekly breakdown
  - Learning outcomes
  - Required tools/hardware
# Module 4: Vision–Language–Action (VLA)
  - Chapters
  - Weekly breakdown
  - Learning outcomes
  - Required tools/hardware
# Capstone Module: Autonomous Humanoid System
  - Tasks
  - Outcomes
  - Required tools/hardware
# Constraints & Notes
  - Enforce constitution rules
  - Respect spec.md rules
  - Maintain module/chapter sequencing

CRITICAL CONSTRAINTS:
- Do NOT generate chapter content
- Do NOT invent new modules or weeks
- Strictly follow constitution.md and spec.md
- Plan must be fully actionable for Claude Code in the next steps

Generate the full book_plan.md now.
```

## Response snapshot

```markdown
# Physical AI & Humanoid Robotics - Book Creation Plan

## Overview

This plan outlines the comprehensive creation of the Physical AI & Humanoid Robotics book, following the module structure defined in the constitution and spec. The book will be created in 5 modules over 16 weeks, with each module building upon the previous ones. Each chapter will focus on specific learning outcomes and required tools, ensuring students develop a complete understanding of embodied AI systems.

The plan strictly follows the constitution's rules:
- Book is divided into exactly five fixed modules that cannot be reordered, merged, or skipped
- Each chapter must declare its parent module
- No content may anticipate or reference future module concepts before their designated weeks
- All AI systems described must have direct physical manifestation and control over robotic hardware or simulation environments

## Module 1: ROS 2 – The Robotic Nervous System

### Chapters

**Chapter 1.1: Introduction to ROS 2 Architecture**
- Week 1
- Learning outcomes: Understand ROS 2 architecture, nodes, topics, and services
- Required tools/hardware: ROS 2 Humble Hawksbill, rclpy, basic Python environment
- Dependencies: None (foundational chapter)

**Chapter 1.2: ROS 2 Nodes, Topics, and Message Passing**
- Week 1
- Learning outcomes: Create and manage ROS 2 nodes, publish/subscribe to topics
- Required tools/hardware: ROS 2 Humble Hawksbill, rclpy, rviz2
- Dependencies: Chapter 1.1

**Chapter 1.3: Services, Actions, and Parameters**
- Week 2
- Learning outcomes: Implement services and actions for synchronous and asynchronous communication
- Required tools/hardware: ROS 2 Humble Hawksbill, rclpy
- Dependencies: Chapter 1.1, Chapter 1.2

**Chapter 1.4: Robot State Publishing and tf2**
- Week 2
- Learning outcomes: Publish robot state information and transform frames
- Required tools/hardware: ROS 2 Humble Hawksbill, rclpy, tf2 libraries
- Dependencies: Chapter 1.1, Chapter 1.2

**Chapter 1.5: Launch Files and System Management**
- Week 3
- Learning outcomes: Organize and launch complex ROS 2 systems
- Required tools/hardware: ROS 2 Humble Hawksbill, launch system
- Dependencies: All previous chapters in Module 1

**Chapter 1.6: ROS 2 Testing and Debugging**
- Week 3
- Learning outcomes: Test and debug ROS 2 applications effectively
- Required tools/hardware: ROS 2 Humble Hawksbill, testing frameworks, rviz2
- Dependencies: All previous chapters in Module 1

### Weekly breakdown
- **Week 1**: Introduction to ROS 2 architecture and basic communication patterns (nodes, topics)
- **Week 2**: Advanced communication patterns (services, actions) and robot state management
- **Week 3**: System organization and debugging techniques

### Learning outcomes
By the end of Module 1, students will be able to:
- Design and implement ROS 2 communication infrastructure
- Create nodes that publish and subscribe to topics
- Implement services and actions for complex communication
- Manage robot state and coordinate transformations
- Organize and launch complex ROS 2 systems
- Test and debug ROS 2 applications

### Required tools/hardware
- ROS 2 Humble Hawksbill
- rclpy Python client library
- rviz2 visualization tool
- Python 3.10+
- Basic computer with Ubuntu 22.04 LTS

## Module 2: Digital Twins – Gazebo & Unity

### Chapters

**Chapter 2.1: Introduction to Simulation Environments**
- Week 4
- Learning outcomes: Understand simulation concepts and setup Gazebo Garden
- Required tools/hardware: Gazebo Garden, ROS 2 Humble Hawksbill, rclpy
- Dependencies: Module 1 completed

**Chapter 2.2: Robot Modeling and URDF Creation**
- Week 4
- Learning outcomes: Create robot models using URDF (Unified Robot Description Format)
- Required tools/hardware: Gazebo Garden, URDF tools, text editor
- Dependencies: Chapter 2.1

**Chapter 2.3: Physics Simulation and Environment Modeling**
- Week 5
- Learning outcomes: Configure physics properties and create simulation environments
- Required tools/hardware: Gazebo Garden, physics engine tools
- Dependencies: Chapter 2.1, Chapter 2.2

**Chapter 2.4: Sensor Simulation and Integration**
- Week 5
- Learning outcomes: Add and configure simulated sensors in Gazebo
- Required tools/hardware: Gazebo Garden, sensor plugins, ROS 2 integration
- Dependencies: Chapter 2.1, Chapter 2.2, Chapter 2.3

**Chapter 2.5: Unity 3D Simulation Environment Setup**
- Week 6
- Learning outcomes: Set up Unity 3D for robotics simulation
- Required tools/hardware: Unity 2022.3 LTS, Unity robotics packages
- Dependencies: Chapter 2.1, Chapter 2.2

**Chapter 2.6: Testing and Validation in Simulated Environments**
- Week 6
- Learning outcomes: Validate robot behaviors in simulation before hardware deployment
- Required tools/hardware: Gazebo Garden, Unity 2022.3 LTS, ROS 2 integration
- Dependencies: All previous chapters in Module 2

### Weekly breakdown
- **Week 4**: Introduction to simulation and robot modeling
- **Week 5**: Physics simulation and sensor integration
- **Week 6**: Alternative simulation platforms and validation techniques

### Learning outcomes
By the end of Module 2, students will be able to:
- Set up and configure Gazebo Garden simulation environment
- Create robot models using URDF format
- Configure physics properties and environment parameters
- Integrate simulated sensors with ROS 2
- Set up Unity 3D for robotics simulation
- Validate robot behaviors in simulation before hardware deployment

### Required tools/hardware
- Gazebo Garden
- Unity 2022.3 LTS
- ROS 2 Humble Hawksbill
- Python 3.10+
- RTX-class GPU for simulation rendering
- URDF tools and modeling utilities

## Module 3: AI Robot Brain – NVIDIA Isaac

### Chapters

**Chapter 3.1: Introduction to NVIDIA Isaac ROS**
- Week 7
- Learning outcomes: Understand NVIDIA Isaac framework and setup
- Required tools/hardware: NVIDIA Isaac ROS, Jetson Orin, CUDA toolkit
- Dependencies: Module 1 and Module 2 completed

**Chapter 3.2: Perception Pipelines with Isaac**
- Week 7
- Learning outcomes: Create perception pipelines for sensor data processing
- Required tools/hardware: NVIDIA Isaac ROS, perception tools, camera sensors
- Dependencies: Chapter 3.1

**Chapter 3.3: Navigation and Path Planning**
- Week 8
- Learning outcomes: Implement navigation systems and path planning algorithms
- Required tools/hardware: NVIDIA Isaac ROS, navigation stack, mapping tools
- Dependencies: Chapter 3.1, Chapter 3.2

**Chapter 3.4: Control Algorithms with Isaac**
- Week 8
- Learning outcomes: Develop control algorithms for robot movement
- Required tools/hardware: NVIDIA Isaac ROS, control libraries
- Dependencies: Chapter 3.1, Chapter 3.2, Chapter 3.3

**Chapter 3.5: AI Processing for Robotic Systems**
- Week 9
- Learning outcomes: Implement AI processing pipelines for robotic decision making
- Required tools/hardware: NVIDIA Isaac ROS, AI frameworks, Jetson Orin
- Dependencies: Chapter 3.1, Chapter 3.2, Chapter 3.3, Chapter 3.4

**Chapter 3.6: Isaac Sim Integration**
- Week 10
- Learning outcomes: Integrate Isaac Sim for advanced simulation
- Required tools/hardware: Isaac Sim, NVIDIA Isaac ROS, simulation tools
- Dependencies: All previous chapters in Module 3

### Weekly breakdown
- **Week 7**: Introduction to Isaac framework and perception pipelines
- **Week 8**: Navigation, path planning, and control algorithms
- **Week 9**: AI processing and decision making
- **Week 10**: Isaac Sim integration and advanced simulation

### Learning outcomes
By the end of Module 3, students will be able to:
- Set up and configure NVIDIA Isaac ROS framework
- Create perception pipelines for sensor data processing
- Implement navigation systems and path planning algorithms
- Develop control algorithms for robot movement
- Implement AI processing pipelines for robotic decision making
- Integrate Isaac Sim for advanced simulation and testing

### Required tools/hardware
- NVIDIA Isaac ROS
- Jetson Orin AGX
- Isaac Sim
- CUDA toolkit
- RTX-class GPU
- Python 3.10+
- Camera and sensor simulation tools

## Module 4: Vision–Language–Action (VLA)

### Chapters

**Chapter 4.1: Introduction to Multimodal AI**
- Week 11
- Learning outcomes: Understand vision-language-action integration concepts
- Required tools/hardware: NVIDIA cuDF, multimodal frameworks, RTX GPU
- Dependencies: Module 1, 2, and 3 completed

**Chapter 4.2: Vision Processing for Robotics**
- Week 11
- Learning outcomes: Process visual information for robotic applications
- Required tools/hardware: Computer vision libraries, image processing tools
- Dependencies: Chapter 4.1

**Chapter 4.3: Language Understanding and Processing**
- Week 12
- Learning outcomes: Integrate natural language understanding for robot commands
- Required tools/hardware: NLP frameworks, speech recognition APIs
- Dependencies: Chapter 4.1, Chapter 4.2

**Chapter 4.4: Voice-to-Action Pipelines**
- Week 12
- Learning outcomes: Create voice command systems for robot control
- Required tools/hardware: Speech recognition APIs, audio processing tools
- Dependencies: Chapter 4.1, Chapter 4.2, Chapter 4.3

**Chapter 4.5: Cognitive Robotics and Planning**
- Week 13
- Learning outcomes: Implement cognitive planning systems for robot behavior
- Required tools/hardware: Planning frameworks, cognitive AI tools
- Dependencies: Chapter 4.1, Chapter 4.2, Chapter 4.3, Chapter 4.4

**Chapter 4.6: Vision-Language Models for Robotic Applications**
- Week 14
- Learning outcomes: Deploy VLA models for autonomous robot behavior
- Required tools/hardware: Multimodal transformers, NVIDIA cuDF, RTX GPU
- Dependencies: All previous chapters in Module 4

### Weekly breakdown
- **Week 11**: Introduction to multimodal AI and vision processing
- **Week 12**: Language understanding and voice-to-action pipelines
- **Week 13**: Cognitive robotics and planning systems
- **Week 14**: Vision-language models and autonomous behavior

### Learning outcomes
By the end of Module 4, students will be able to:
- Understand and implement vision-language-action integration
- Process visual information for robotic applications
- Integrate natural language understanding for robot commands
- Create voice command systems for robot control
- Implement cognitive planning systems for robot behavior
- Deploy VLA models for autonomous robot behavior

### Required tools/hardware
- NVIDIA cuDF
- Multimodal transformers
- Speech recognition APIs
- RTX-class GPU
- Computer vision libraries
- NLP frameworks
- Python 3.10+

## Capstone Module: Autonomous Humanoid System

### Tasks

**Task 5.1: System Architecture Design**
- Week 15
- Outcome: Complete system architecture integrating all previous modules
- Required tools/hardware: Design tools, all previous module components

**Task 5.2: Integration of ROS 2 Infrastructure**
- Week 15
- Outcome: Unified ROS 2 communication system across all components
- Required tools/hardware: ROS 2 Humble Hawksbill, rclpy, integration tools

**Task 5.3: Simulation-to-Real Mapping**
- Week 16
- Outcome: System validated in simulation and deployed on physical hardware
- Required tools/hardware: Gazebo Garden, Isaac Sim, physical humanoid platform

**Task 5.4: Performance Optimization and Testing**
- Week 16
- Outcome: System meeting real-time performance requirements
- Required tools/hardware: Performance profiling tools, RTX GPU, Jetson Orin

### Outcomes
By the end of the Capstone Module, students will have:
- Integrated all modules into a complete autonomous humanoid system
- Demonstrated the system operating in both simulation and physical environments
- Verified real-time performance with latency constraints <100ms
- Implemented multimodal AI capabilities (vision-language-action) for autonomous behavior
- Applied hardware-software co-design principles

### Required tools/hardware
- Full stack from previous modules
- Physical or simulated humanoid platform
- RTX-class GPU
- NVIDIA Jetson Orin AGX
- Performance profiling tools
- All tools from previous modules

## Constraints & Notes

### Enforce constitution rules
- Each chapter must declare its parent module at the beginning
- No content may anticipate or reference future module concepts before their designated weeks
- All AI systems described must have direct physical manifestation and control over robotic hardware or simulation environments
- All code must use ROS 2 Python APIs (rclpy) or NVIDIA Isaac frameworks
- Python version must be 3.10 or higher
- All explanations must be physically grounded with clear mapping between simulation and real hardware
- Marketing language and buzzwords are strictly forbidden
- Purely digital AI without physical embodiment is forbidden

### Respect spec.md rules
- Content must be precise, technical, and focused on implementation
- Each concept must be explained with specific code examples, hardware specifications, or simulation parameters
- All examples must run on simulation environments before hardware deployment
- Simulation code must be clearly separated from hardware code
- All code examples must be testable and complete
- ROS 2 terminology must follow official documentation exactly
- NVIDIA Isaac components must use official naming conventions

### Maintain module/chapter sequencing
- Content depth must follow the weekly breakdown sequence: Weeks 1-3 for Module 1, Weeks 4-6 for Module 2, Weeks 7-10 for Module 3, Weeks 11-14 for Module 4, and Weeks 15-16 for Capstone
- Each week's content must build incrementally on previous weeks without skipping ahead
- No chapter may span multiple modules or exist outside the module system
- The module structure is immutable and all content must fit within this framework
```

## Outcome

- ✅ Impact: Created comprehensive book creation plan for Physical AI & Humanoid Robotics
- 🧪 Tests: N/A
- 📁 Files: Created book_plan.md with complete module structure
- 🔁 Next prompts: Ready for task generation with /sp.tasks
- 🧠 Reflection: Plan follows all required sections and respects constitution/spec constraints

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A