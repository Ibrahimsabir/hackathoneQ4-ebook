---
id: 1
title: Create Physical AI Constitution
stage: constitution
date: 2025-12-23
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,physical-ai,robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

```text
You are a senior curriculum architect, AI systems designer, and technical constitution author.

Your task is to generate a STRICT and ENFORCEABLE "constitution.md" file
for a Spec-Kit Plus driven technical book on
Physical AI & Humanoid Robotics.

You have been provided with detailed course content including:
- Course goals and philosophy
- Clearly defined modules
- Weekly breakdown
- Learning outcomes
- Hardware, simulation, and lab architecture
- Cloud vs on-prem execution constraints
- Capstone project definition

YOUR OBJECTIVE:
Extract NON-NEGOTIABLE RULES and STRUCTURAL CONSTRAINTS from the content
and convert them into a "constitution.md" file that will govern ALL future
AI-written chapters of the book.

This constitution must ensure:
- The book is MODULE-DRIVEN
- Each chapter belongs to EXACTLY ONE module
- No topic crosses module boundaries
- Learning progression follows the defined order
- Physical AI constraints are always respected

--------------------------------------------------
OUTPUT FILE REQUIREMENTS
--------------------------------------------------
- Output ONE file only: constitution.md
- Spec-Kit Plus compatible
- Use Markdown headings
- Rules must be imperative and enforceable
- No summaries, no teaching, no explanations
- No deployment or DevOps content

--------------------------------------------------
MANDATORY SECTIONS (IN THIS ORDER)
--------------------------------------------------

# Purpose of the Constitution
Define what kind of book this is and what kind of AI systems it is allowed to describe.

# Target Audience Constraints
Define assumed background, forbidden assumptions, and learner responsibilities.

# Book Structure & Module System
Define the REQUIRED module-based structure of the book.
Rules must specify that:
- The book is divided into fixed modules
- Modules cannot be reordered, merged, or skipped
- Each chapter must declare its parent module

# Module Definitions & Boundaries
Explicitly define EACH module with:
- Its purpose
- Allowed topics
- Forbidden topics
- Required tools and platforms

Modules to enforce:
- Module 1: ROS 2 – The Robotic Nervous System
- Module 2: Digital Twins – Gazebo & Unity
- Module 3: AI Robot Brain – NVIDIA Isaac
- Module 4: Vision–Language–Action (VLA)
- Capstone Module: Autonomous Humanoid System

# Weekly Progression Rules
Define how weekly breakdown constrains depth and pacing.
No future content may violate the weekly sequencing logic.

# Writing Style Rules
Rules governing tone, density, precision, and technical depth.

# Explanation Rules
Rules enforcing:
- Physical grounding
- System-level reasoning
- Simulation-to-real mapping
- Cause–effect explanations

# Code & Technical Content Rules
Rules governing:
- ROS 2 code
- Python usage
- Real APIs only
- Simulation vs hardware code separation

# Hardware & Reality Constraints
Rules enforcing:
- RTX GPU requirements
- Jetson resource limits
- Latency constraints
- Physics realism

# AI & Robotics Integration Rules
Rules governing:
- LLM usage
- Voice-to-action pipelines
- Cognitive planning vs execution
- VLA system boundaries

# Examples, Labs & Capstone Rules
Rules for exercises, labs, and the final project.

# Terminology & Naming Rules
Rules to maintain consistency across modules and chapters.

# What Is Strictly Forbidden
Hard bans on:
- Marketing language
- Purely digital AI without embodiment
- Hand-wavy cloud assumptions
- Ignoring hardware limits

# AI Behavior Rules
Rules that the AI writer must obey at all times.

--------------------------------------------------
CRITICAL CONSTRAINTS
--------------------------------------------------
- DO NOT rewrite or summarize the content
- DO NOT invent modules, tools, or weeks
- DO NOT soften rules into suggestions
- Extract rules ONLY from the provided content
- Write as if this constitution will be used to audit AI output

Generate the constitution.md file now.
```

## Response snapshot

```markdown
<!--
Sync Impact Report:
Version change: N/A → 1.0.0
List of modified principles: N/A (new constitution)
Added sections: All sections (new constitution file)
Removed sections: N/A
Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Constitution

## Purpose of the Constitution

This constitution governs a technical book on Physical AI & Humanoid Robotics. The book describes embodied AI systems that integrate with physical robots, specifically humanoid platforms. All AI systems described must have direct physical manifestation and control over robotic hardware or simulation environments. The book must focus exclusively on AI systems that interact with the physical world through robotic embodiment.

## Target Audience Constraints

The target audience has intermediate programming skills in Python and basic understanding of robotics concepts. The book must not assume cloud computing expertise, advanced mathematics beyond calculus, or specialized hardware knowledge. Learners must have access to RTX-class GPUs or Jetson platforms and be willing to work with simulation environments. The book must not assume pure software/AI background without robotics context.

## Book Structure & Module System

The book is divided into exactly five fixed modules that cannot be reordered, merged, or skipped. Each chapter must declare its parent module at the beginning. The module structure is immutable and all content must fit within this framework. No chapter may span multiple modules or exist outside the module system.

## Module Definitions & Boundaries

### Module 1: ROS 2 – The Robotic Nervous System

Purpose: Establish the communication and control infrastructure for robotic systems. Allowed topics include ROS 2 nodes, topics, services, actions, and message passing. Forbidden topics include non-ROS communication protocols, direct hardware interfaces without ROS, or cloud-only architectures. Required tools: ROS 2 Humble Hawksbill, rclpy, rviz2.

### Module 2: Digital Twins – Gazebo & Unity

Purpose: Create simulation environments for robot testing and development. Allowed topics include Gazebo physics simulation, Unity 3D environments, sensor simulation, and environment modeling. Forbidden topics include pure physical robot work without simulation, non-physics-based simulators, or cloud-based simulation platforms. Required tools: Gazebo Garden, Unity 2022.3 LTS.

### Module 3: AI Robot Brain – NVIDIA Isaac

Purpose: Implement AI processing and decision-making for robotic systems. Allowed topics include NVIDIA Isaac ROS, perception pipelines, navigation, and control algorithms. Forbidden topics include non-NVIDIA AI platforms, cloud-only AI processing, or pure computer vision without robotic application. Required tools: NVIDIA Isaac ROS, Isaac Sim, Jetson Orin.

### Module 4: Vision–Language–Action (VLA)

Purpose: Integrate multimodal AI for robot understanding and execution. Allowed topics include vision-language models, action planning, voice-to-action pipelines, and cognitive robotics. Forbidden topics include single-modal AI systems, text-only LLMs without vision/action components, or non-robotic applications. Required tools: NVIDIA cuDF, multimodal transformers, speech recognition APIs.

### Capstone Module: Autonomous Humanoid System

Purpose: Integrate all previous modules into a complete humanoid robot system. Allowed topics include system integration, real-time performance, hardware-software co-design, and complete autonomous behaviors. Forbidden topics include isolated component testing, non-humanoid robots, or partial system demonstrations. Required tools: Full stack from previous modules, physical or simulated humanoid platform.

## Weekly Progression Rules

Content depth must follow the weekly breakdown sequence: Weeks 1-3 for Module 1, Weeks 4-6 for Module 2, Weeks 7-10 for Module 3, Weeks 11-14 for Module 4, and Weeks 15-16 for Capstone. No content may anticipate or reference future module concepts before their designated weeks. Each week's content must build incrementally on previous weeks without skipping ahead.

## Writing Style Rules

The writing must be precise, technical, and focused on implementation. Each concept must be explained with specific code examples, hardware specifications, or simulation parameters. No marketing language, buzzwords, or abstract concepts without concrete implementation details. Density must balance accessibility with technical depth, providing sufficient detail for implementation without excessive background information.

## Explanation Rules

All explanations must be physically grounded with clear mapping between simulation and real hardware. System-level reasoning must show how components interact as a whole. Simulation-to-real mapping must be explicitly described for every concept. Cause-effect relationships must be clearly articulated with specific technical mechanisms, not abstract relationships.

## Code & Technical Content Rules

All code must use ROS 2 Python APIs (rclpy) or NVIDIA Isaac frameworks. Python version must be 3.10 or higher. All APIs referenced must be real, documented APIs with specific version numbers. Simulation code must be clearly separated from hardware code with conditional compilation or clear environment detection. All code examples must be testable and complete.

## Hardware & Reality Constraints

All systems must be designed for RTX-class GPUs or NVIDIA Jetson platforms. Jetson resource limits must be explicitly considered with memory and compute constraints. Latency constraints must be specified for real-time systems (typically <100ms for control loops). Physics realism must be maintained with accurate simulation parameters that match real-world physics.

## AI & Robotics Integration Rules

LLM usage must be constrained to multimodal (vision-language-action) applications with clear robotic embodiment. Voice-to-action pipelines must include specific audio processing and natural language understanding components. Cognitive planning must be clearly separated from low-level execution with defined interfaces. VLA system boundaries must specify input/output formats and processing constraints.

## Examples, Labs & Capstone Rules

All examples must run on simulation environments before hardware deployment. Labs must include both simulation and hardware validation steps. The capstone project must integrate all modules into a complete autonomous humanoid system. All exercises must include success criteria with measurable performance metrics.

## Terminology & Naming Rules

ROS 2 terminology must follow official documentation exactly. NVIDIA Isaac components must use official naming conventions. Hardware platform names must be specific (e.g., "NVIDIA Jetson Orin AGX" not "Jetson platform"). Simulation environment names must be precise (e.g., "Gazebo Garden" not "Gazebo simulator").

## What Is Strictly Forbidden

Marketing language and buzzwords are strictly forbidden. Purely digital AI without physical embodiment is forbidden. Hand-wavy cloud assumptions without specific hardware constraints are forbidden. Ignoring hardware limits and resource constraints is forbidden. Pure software AI tutorials without robotic application are forbidden.

## AI Behavior Rules

The AI writer must always consider physical embodiment when describing AI systems. All recommendations must include hardware resource implications. All code examples must be testable in simulation before hardware deployment. The AI must respect module boundaries and not cross-contaminate topics between modules. All technical claims must be verifiable with specific implementation details.

## Governance

This constitution supersedes all other practices and guidelines for the Physical AI & Humanoid Robotics book. All chapters, code examples, and exercises must comply with these rules. Amendments require explicit documentation of changes, approval from curriculum architects, and migration plans for existing content. All reviews must verify constitution compliance. Complexity must be justified with specific hardware or performance requirements.

**Version**: 1.0.0 | **Ratified**: 2025-12-23 | **Last Amended**: 2025-12-23
```

## Outcome

- ✅ Impact: Created comprehensive constitution file for Physical AI & Humanoid Robotics book
- 🧪 Tests: N/A
- 📁 Files: Created .specify/memory/constitution.md with module-based structure
- 🔁 Next prompts: N/A
- 🧠 Reflection: Constitution enforces strict module boundaries and physical AI focus

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A