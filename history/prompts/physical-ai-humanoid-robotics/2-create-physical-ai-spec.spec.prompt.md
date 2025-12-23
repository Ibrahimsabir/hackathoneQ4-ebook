---
id: 2
title: Create Physical AI Spec
stage: spec
date: 2025-12-23
surface: agent
model: Claude Sonnet 4.5
feature: physical-ai-humanoid-robotics
branch: 1-physical-ai-humanoid-robotics
user: user
command: /sp.specify
labels: [spec,physical-ai,robotics,curriculum]
links:
  spec: specs/1-physical-ai-humanoid-robotics/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-physical-ai-humanoid-robotics/spec.md
tests:
---

## Prompt

```text
You are a senior AI curriculum architect and Spec-Kit Plus specification designer.

Your task is to create a complete Spec-Kit Plus specification file (spec.md)
for a Physical AI & Humanoid Robotics book/course.

INPUT:
- Raw course content (provided)
- constitution.md (already generated, strict rules enforced)

YOUR OBJECTIVE:
Generate a Spec-Kit Plus specification that will guide Claude Code
to write the book chapter-by-chapter, module-by-module, week-by-week.

The specification must include:
1. Vision: Purpose of the book, learning philosophy, target audience
2. Audience: Skill level, prerequisites, assumed knowledge
3. Outline: Module structure, chapters, weekly breakdown
4. Rules: Writing rules, explanation style, code style, terminology
5. Tasks: High-level tasks (module/chapter/week) that will later guide chapter generation
6. Constraints: Hardware, simulation, physical AI limits, cloud vs on-prem
7. Capstone: Final project requirements

OUTPUT REQUIREMENTS:
- Output ONE file: spec.md
- Spec-Kit Plus compatible
- Use clear headings
- Use bullet points for rules and constraints
- Be strict, enforceable, and curriculum-ready
- Do not generate actual chapter content
- Respect constitution.md fully

MANDATORY SECTIONS:
# Vision
# Audience
# Outline
# Rules
# Constraints
# Capstone / Project Guidelines
# Notes for AI Writer

CRITICAL CONSTRAINTS:
- Do not invent modules or weeks
- Do not summarize content; extract structure and rules
- All AI chapter outputs MUST follow this spec

Generate the Spec-Kit Plus specification file now.
```

## Response snapshot

```markdown
# Feature Specification: Physical AI & Humanoid Robotics

**Feature Branch**: `1-physical-ai-humanoid-robotics`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "You are a senior AI curriculum architect and Spec-Kit Plus specification designer.

Your task is to create a complete Spec-Kit Plus specification file (spec.md)
for a Physical AI & Humanoid Robotics book/course.

INPUT:
- Raw course content (provided)
- constitution.md (already generated, strict rules enforced)

YOUR OBJECTIVE:
Generate a Spec-Kit Plus specification that will guide Claude Code
to write the book chapter-by-chapter, module-by-module, week-by-week.

The specification must include:
1. Vision: Purpose of the book, learning philosophy, target audience
2. Audience: Skill level, prerequisites, assumed knowledge
3. Outline: Module structure, chapters, weekly breakdown
4. Rules: Writing rules, explanation style, code style, terminology
5. Tasks: High-level tasks (module/chapter/week) that will later guide chapter generation
6. Constraints: Hardware, simulation, physical AI limits, cloud vs on-prem
7. Capstone: Final project requirements

OUTPUT REQUIREMENTS:
- Output ONE file: spec.md
- Spec-Kit Plus compatible
- Use clear headings
- Use bullet points for rules and constraints
- Be strict, enforceable, and curriculum-ready
- Do not generate actual chapter content
- Respect constitution.md fully

MANDATORY SECTIONS:
# Vision
# Audience
# Outline
# Rules
# Constraints
# Capstone / Project Guidelines
# Notes for AI Writer

CRITICAL CONSTRAINTS:
- Do not invent modules or weeks
- Do not summarize content; extract structure and rules
- All AI chapter outputs MUST follow this spec

Generate the Spec-Kit Plus specification file now."

## Vision

This Physical AI & Humanoid Robotics book is designed to teach readers how to build AI systems that are embodied in physical robotic platforms, specifically humanoid robots. The book emphasizes the integration of AI with physical reality through simulation and real-world robotics applications. The learning philosophy centers on hands-on implementation with a progression from basic robotic communication systems to complex multimodal AI that enables autonomous humanoid behavior. The target audience includes robotics engineers, AI practitioners, and computer science students who want to understand how to create intelligent systems that interact with the physical world through robotic embodiment.

## Audience

The target audience has intermediate programming skills in Python and basic understanding of robotics concepts. Learners should have calculus-level mathematics knowledge and access to RTX-class GPUs or NVIDIA Jetson platforms. The book assumes familiarity with basic programming concepts but does not require expertise in cloud computing, advanced mathematics beyond calculus, or specialized hardware knowledge. Students must be willing to work with simulation environments and have access to appropriate hardware for practical exercises.

## Outline

The book is structured into five distinct modules that build upon each other:

**Module 1: ROS 2 – The Robotic Nervous System (Weeks 1-3)**
- Introduction to ROS 2 communication infrastructure
- ROS 2 nodes, topics, services, and actions
- Robot state publishing and visualization

**Module 2: Digital Twins – Gazebo & Unity (Weeks 4-6)**
- Simulation environment creation and physics modeling
- Sensor simulation and environment modeling
- Testing and validation in simulated environments

**Module 3: AI Robot Brain – NVIDIA Isaac (Weeks 7-10)**
- NVIDIA Isaac ROS frameworks and perception pipelines
- Navigation and control algorithms
- AI processing for robotic systems

**Module 4: Vision–Language–Action (VLA) (Weeks 11-14)**
- Multimodal AI integration (vision, language, action)
- Voice-to-action pipelines and cognitive robotics
- Vision-language models for robotic applications

**Capstone Module: Autonomous Humanoid System (Weeks 15-16)**
- Integration of all previous modules
- Complete autonomous humanoid system implementation
- Real-time performance and hardware-software co-design

## Rules

### Writing Style Rules
- Content must be precise, technical, and focused on implementation
- Each concept must be explained with specific code examples, hardware specifications, or simulation parameters
- No marketing language, buzzwords, or abstract concepts without concrete implementation details
- Density must balance accessibility with technical depth

### Explanation Rules
- All explanations must be physically grounded with clear mapping between simulation and real hardware
- System-level reasoning must show how components interact as a whole
- Simulation-to-real mapping must be explicitly described for every concept
- Cause-effect relationships must be clearly articulated with specific technical mechanisms

### Code Style Rules
- All code must use ROS 2 Python APIs (rclpy) or NVIDIA Isaac frameworks
- Python version must be 3.10 or higher
- All APIs referenced must be real, documented APIs with specific version numbers
- Simulation code must be clearly separated from hardware code

### Terminology Rules
- ROS 2 terminology must follow official documentation exactly
- NVIDIA Isaac components must use official naming conventions
- Hardware platform names must be specific (e.g., "NVIDIA Jetson Orin AGX" not "Jetson platform")
- Simulation environment names must be precise (e.g., "Gazebo Garden" not "Gazebo simulator")

## Constraints

### Hardware Constraints
- All systems must be designed for RTX-class GPUs or NVIDIA Jetson platforms
- Jetson resource limits must be explicitly considered with memory and compute constraints
- Latency constraints must be specified for real-time systems (typically <100ms for control loops)
- Physics realism must be maintained with accurate simulation parameters

### Simulation vs Hardware
- All examples must run on simulation environments before hardware deployment
- Simulation code must be clearly separated from hardware code
- Hardware-specific implementations must be conditional or clearly marked

### Physical AI Requirements
- All AI systems described must have direct physical manifestation and control over robotic hardware or simulation environments
- Content must focus exclusively on AI systems that interact with the physical world through robotic embodiment
- Purely digital AI without physical embodiment is forbidden

### Cloud vs On-Prem
- Systems must be designed for on-premise deployment with local GPU resources
- Cloud-only architectures are forbidden
- Real-time performance requirements preclude cloud-based processing for control loops

## Capstone / Project Guidelines

The capstone project must integrate all modules into a complete autonomous humanoid system. Students must demonstrate:
- Integration of ROS 2 communication infrastructure with perception and control
- Successful deployment in both simulation and physical environments
- Real-time performance with specified latency constraints
- Multimodal AI capabilities (vision-language-action) for autonomous behavior
- Hardware-software co-design principles

The project must include measurable performance metrics, success criteria with quantifiable outcomes, and clear demonstration of autonomous behavior. Students must show the system operating continuously for a specified duration and completing predefined tasks in both simulation and physical environments.

## Notes for AI Writer

- Always consider physical embodiment when describing AI systems
- All recommendations must include hardware resource implications
- All code examples must be testable in simulation before hardware deployment
- Respect module boundaries and do not cross-contaminate topics between modules
- All technical claims must be verifiable with specific implementation details
- Follow the weekly progression sequence without skipping ahead
- Maintain strict adherence to the constitution.md rules
- Ensure each chapter declares its parent module
- No content may anticipate or reference future module concepts before their designated weeks
```

## Outcome

- ✅ Impact: Created comprehensive spec file for Physical AI & Humanoid Robotics book
- 🧪 Tests: N/A
- 📁 Files: Created specs/1-physical-ai-humanoid-robotics/spec.md with complete structure
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Spec follows all required sections and respects constitution constraints

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A