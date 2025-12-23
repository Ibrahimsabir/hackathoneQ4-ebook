# Module 4: Vision–Language–Action (VLA)

## Chapter 4.5: Cognitive Robotics and Planning

This chapter explores cognitive robotics and planning systems that enable robots to think, reason, and plan complex behaviors. Cognitive robotics combines perception, reasoning, and action to create intelligent systems capable of complex autonomous behavior.

### Understanding Cognitive Robotics

Cognitive robotics focuses on creating robots that exhibit cognitive behaviors similar to biological systems. Key aspects include:

- **Perception**: Understanding the environment through sensors
- **Reasoning**: Making logical inferences from perceived information
- **Planning**: Developing sequences of actions to achieve goals
- **Learning**: Adapting behavior based on experience
- **Memory**: Storing and retrieving information about past experiences
- **Decision Making**: Choosing appropriate actions based on current state and goals

### Cognitive Architecture Components

The cognitive robotics architecture includes:

```
+-------------------+
|   Decision Maker  |
|   (Goal Selection)|
+-------------------+
|   Planner         |
|   (Action Sequences)|
+-------------------+
|   Reasoner        |
|   (Logical Inference)|
+-------------------+
|   Memory System   |
|   (Knowledge Base)|
+-------------------+
|   Perception      |
|   (Sensory Input) |
+-------------------+
```

### Basic Cognitive Architecture

Implementing a foundational cognitive architecture:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass
from enum import Enum

class CognitiveState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"

@dataclass
class WorldState:
    """Represents the current state of the world"""
    objects: List[Dict] = None
    robot_pose: Pose = None
    environment_map: np.ndarray = None
    goals: List[Dict] = None
    obstacles: List[Dict] = None
    recent_events: List[Dict] = None

@dataclass
class PlanStep:
    """Represents a single step in a plan"""
    action: str
    parameters: Dict
    priority: int
    duration: float
    preconditions: List[str]
    effects: List[str]

class CognitiveRobot(Node):
    def __init__(self):
        super().__init__('cognitive_robot')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            Pose,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/cognitive_status',
            10
        )

        self.plan_pub = self.create_publisher(
            String,
            '/cognitive_plan',
            10
        )

        # Initialize cognitive components
        self.perception_module = None
        self.reasoning_module = None
        self.planning_module = None
        self.memory_system = None
        self.initialize_cognitive_modules()

        # State variables
        self.current_state = WorldState()
        self.current_plan = []
        self.current_goal = None
        self.cognitive_state = CognitiveState.IDLE
        self.plan_index = 0
        self.last_action_time = time.time()

        # Memory and learning
        self.experience_buffer = []
        self.behavior_history = []

        # Control timer
        self.cognitive_timer = self.create_timer(0.1, self.cognitive_processing_loop)

        self.get_logger().info('Cognitive robot initialized')

    def initialize_cognitive_modules(self):
        """Initialize cognitive processing modules"""
        try:
            # Perception module
            class PerceptionModule:
                def __init__(self, node):
                    self.node = node
                    self.object_tracker = None
                    self.scene_analyzer = None
                    self.initialize_components()

                def initialize_components(self):
                    """Initialize perception components"""
                    # Object tracking
                    class SimpleObjectTracker:
                        def __init__(self):
                            self.objects = {}
                            self.next_id = 0

                        def update_objects(self, detections):
                            """Update object tracking with new detections"""
                            for detection in detections:
                                # In a real implementation, this would use Kalman filters or similar
                                obj_id = self.next_id
                                self.next_id += 1

                                obj_info = {
                                    'id': obj_id,
                                    'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                                    'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                                    'position': {
                                        'x': detection.bbox.center.x,
                                        'y': detection.bbox.center.y
                                    },
                                    'bbox': {
                                        'size_x': detection.bbox.size_x,
                                        'size_y': detection.bbox.size_y
                                    },
                                    'last_seen': time.time()
                                }

                                self.objects[obj_id] = obj_info

                            # Remove old objects (simple timeout)
                            current_time = time.time()
                            expired_objects = [
                                obj_id for obj_id, obj in self.objects.items()
                                if current_time - obj['last_seen'] > 5.0  # 5 seconds
                            ]

                            for obj_id in expired_objects:
                                del self.objects[obj_id]

                            return list(self.objects.values())

                    self.object_tracker = SimpleObjectTracker()

                    # Scene analyzer
                    class SceneAnalyzer:
                        def analyze_scene(self, objects, robot_pose, map_data):
                            """Analyze scene for relevant information"""
                            analysis = {
                                'object_types': {},
                                'spatial_relationships': [],
                                'potential_interactions': [],
                                'safety_factors': []
                            }

                            # Count object types
                            for obj in objects:
                                obj_class = obj.get('class', 'unknown')
                                if obj_class in analysis['object_types']:
                                    analysis['object_types'][obj_class] += 1
                                else:
                                    analysis['object_types'][obj_class] = 1

                            # Analyze spatial relationships
                            if robot_pose:
                                for obj in objects:
                                    dx = obj['position']['x'] - robot_pose.position.x
                                    dy = obj['position']['y'] - robot_pose.position.y
                                    distance = np.sqrt(dx*dx + dy*dy)

                                    relationship = {
                                        'object_id': obj['id'],
                                        'object_class': obj['class'],
                                        'distance': distance,
                                        'angle': np.arctan2(dy, dx)
                                    }
                                    analysis['spatial_relationships'].append(relationship)

                            return analysis

                    self.scene_analyzer = SceneAnalyzer()

                def process_perception(self, image, scan, detections, map_data, robot_pose):
                    """Process all perception inputs and update world state"""
                    # Update object tracking
                    tracked_objects = self.object_tracker.update_objects(detections) if detections else []

                    # Analyze scene
                    scene_analysis = self.scene_analyzer.analyze_scene(
                        tracked_objects, robot_pose, map_data
                    )

                    # Create world state update
                    world_update = {
                        'objects': tracked_objects,
                        'scene_analysis': scene_analysis,
                        'environment_map': map_data,
                        'timestamp': time.time()
                    }

                    return world_update

            # Reasoning module
            class ReasoningModule:
                def __init__(self, node):
                    self.node = node
                    self.rule_engine = None
                    self.initialize_reasoning_engine()

                def initialize_reasoning_engine(self):
                    """Initialize rule-based reasoning engine"""
                    # Simple rule engine for robot reasoning
                    class RuleEngine:
                        def __init__(self):
                            self.rules = [
                                # Navigation rules
                                {
                                    'condition': lambda state: state.get('closest_object', {}).get('distance', float('inf')) < 0.5,
                                    'action': 'avoid_object',
                                    'priority': 10
                                },
                                {
                                    'condition': lambda state: state.get('goal_distance', float('inf')) > 1.0,
                                    'action': 'move_towards_goal',
                                    'priority': 8
                                },
                                {
                                    'condition': lambda state: state.get('object_types', {}).get('person', 0) > 0,
                                    'action': 'follow_person',
                                    'priority': 9
                                },
                                {
                                    'condition': lambda state: state.get('obstacle_ahead', False),
                                    'action': 'avoid_obstacle',
                                    'priority': 11
                                },
                                # Safety rules
                                {
                                    'condition': lambda state: state.get('safety_risk', 0) > 0.8,
                                    'action': 'stop_and_assess',
                                    'priority': 12
                                }
                            ]

                        def evaluate_rules(self, world_state):
                            """Evaluate rules against world state"""
                            applicable_rules = []

                            for rule in self.rules:
                                if rule['condition'](world_state):
                                    applicable_rules.append({
                                        'action': rule['action'],
                                        'priority': rule['priority']
                                    })

                            # Sort by priority (highest first)
                            applicable_rules.sort(key=lambda x: x['priority'], reverse=True)

                            return applicable_rules

                    self.rule_engine = RuleEngine()

                def reason_about_world(self, world_state):
                    """Apply reasoning to world state to determine appropriate actions"""
                    # Evaluate rules
                    applicable_rules = self.rule_engine.evaluate_rules(world_state)

                    # Generate reasoning results
                    reasoning_result = {
                        'applicable_rules': applicable_rules,
                        'primary_intent': applicable_rules[0]['action'] if applicable_rules else 'idle',
                        'confidence': 0.8,  # Default confidence
                        'reasoning_trace': []  # Trace of reasoning steps
                    }

                    return reasoning_result

            # Planning module
            class PlanningModule:
                def __init__(self, node):
                    self.node = node
                    self.path_planner = None
                    self.action_planner = None
                    self.initialize_planners()

                def initialize_planners(self):
                    """Initialize path and action planners"""
                    # Simple path planner
                    class PathPlanner:
                        def plan_path(self, start_pose, goal_pose, map_data):
                            """Simple path planning (in real implementation, use A*, RRT, etc.)"""
                            path = []

                            # Simple straight-line path with obstacle avoidance
                            dx = goal_pose.position.x - start_pose.position.x
                            dy = goal_pose.position.y - start_pose.position.y
                            distance = np.sqrt(dx*dx + dy*dy)

                            if distance > 0.1:  # If goal is not too close
                                steps = max(10, int(distance * 10))  # 10 steps per meter

                                for i in range(steps + 1):
                                    t = i / steps
                                    intermediate_pose = Pose()
                                    intermediate_pose.position.x = start_pose.position.x + t * dx
                                    intermediate_pose.position.y = start_pose.position.y + t * dy
                                    intermediate_pose.orientation = start_pose.orientation  # Keep current orientation

                                    path.append(intermediate_pose)

                            return path

                    # Action planner
                    class ActionPlanner:
                        def plan_actions(self, goal, current_state):
                            """Plan sequence of actions to achieve goal"""
                            actions = []

                            if goal.get('type') == 'approach_object':
                                # Plan approach to object
                                obj_id = goal.get('object_id')
                                if obj_id:
                                    actions.append({
                                        'action': 'move_towards_object',
                                        'parameters': {'object_id': obj_id},
                                        'duration': 2.0,
                                        'priority': 1
                                    })
                                    actions.append({
                                        'action': 'stop_near_object',
                                        'parameters': {'object_id': obj_id, 'distance': 0.3},
                                        'duration': 1.0,
                                        'priority': 2
                                    })

                            elif goal.get('type') == 'navigate_to_location':
                                # Plan navigation to location
                                target_pose = goal.get('pose')
                                if target_pose:
                                    actions.append({
                                        'action': 'navigate_to_pose',
                                        'parameters': {'pose': target_pose},
                                        'duration': 5.0,
                                        'priority': 1
                                    })

                            elif goal.get('type') == 'follow_object':
                                # Plan following behavior
                                obj_id = goal.get('object_id')
                                if obj_id:
                                    actions.append({
                                        'action': 'start_tracking_object',
                                        'parameters': {'object_id': obj_id},
                                        'duration': 0.1,
                                        'priority': 1
                                    })
                                    actions.append({
                                        'action': 'maintain_distance',
                                        'parameters': {'object_id': obj_id, 'distance': 1.0},
                                        'duration': float('inf'),  # Continuous
                                        'priority': 2
                                    })

                            return actions

                    self.path_planner = PathPlanner()
                    self.action_planner = ActionPlanner()

                def plan_behavior(self, goal, current_state):
                    """Plan behavior to achieve goal"""
                    # Plan actions
                    action_plan = self.action_planner.plan_actions(goal, current_state)

                    # Plan path if navigation is involved
                    if goal.get('type') == 'navigate_to_location':
                        path = self.path_planner.plan_path(
                            current_state.get('robot_pose'),
                            goal.get('pose'),
                            current_state.get('environment_map')
                        )
                        action_plan.insert(0, {
                            'action': 'execute_path',
                            'parameters': {'path': path},
                            'duration': len(path) * 0.5,  # 0.5 seconds per step
                            'priority': 0
                        })

                    return action_plan

            # Memory system
            class MemorySystem:
                def __init__(self, node):
                    self.node = node
                    self.episodic_memory = []  # Past experiences
                    self.semantic_memory = {}  # General knowledge
                    self.procedural_memory = {}  # How-to knowledge

                def store_episode(self, experience):
                    """Store an episode in episodic memory"""
                    episode = {
                        'timestamp': time.time(),
                        'state_before': experience.get('state_before'),
                        'action_taken': experience.get('action_taken'),
                        'state_after': experience.get('state_after'),
                        'outcome': experience.get('outcome'),
                        'reward': experience.get('reward', 0.0)
                    }

                    self.episodic_memory.append(episode)

                    # Keep only recent episodes (last 1000)
                    if len(self.episodic_memory) > 1000:
                        self.episodic_memory = self.episodic_memory[-1000:]

                def retrieve_similar_episodes(self, current_state, max_episodes=5):
                    """Retrieve similar past episodes"""
                    # Simple similarity based on object types in scene
                    similar_episodes = []

                    for episode in self.episodic_memory[-50:]:  # Check recent episodes
                        # Calculate similarity score (simplified)
                        similarity = 0.0

                        # Compare object types
                        if (episode['state_before'].get('scene_analysis', {}).get('object_types') and
                            current_state.get('scene_analysis', {}).get('object_types')):
                            prev_objects = episode['state_before']['scene_analysis']['object_types']
                            curr_objects = current_state['scene_analysis']['object_types']

                            # Calculate overlap in object types
                            common_objects = set(prev_objects.keys()) & set(curr_objects.keys())
                            if common_objects:
                                similarity += len(common_objects) * 0.5

                        if similarity > 0.3:  # Threshold for similarity
                            similar_episodes.append({
                                'episode': episode,
                                'similarity': similarity
                            })

                    # Sort by similarity and return top matches
                    similar_episodes.sort(key=lambda x: x['similarity'], reverse=True)
                    return similar_episodes[:max_episodes]

                def learn_from_experience(self, experience):
                    """Learn from experience and update knowledge"""
                    # Update semantic memory with patterns observed
                    state_before = experience.get('state_before', {})
                    action_taken = experience.get('action_taken', {})
                    outcome = experience.get('outcome', {})

                    # Example: Learn about object affordances
                    if (state_before.get('scene_analysis', {}).get('object_types', {}).get('box') and
                        action_taken.get('action') == 'approach_object' and
                        outcome.get('success')):
                        # Learn that boxes can be approached
                        if 'object_affordances' not in self.semantic_memory:
                            self.semantic_memory['object_affordances'] = {}

                        if 'box' not in self.semantic_memory['object_affordances']:
                            self.semantic_memory['object_affordances']['box'] = []

                        if 'approach' not in self.semantic_memory['object_affordances']['box']:
                            self.semantic_memory['object_affordances']['box'].append('approach')

            # Initialize modules
            self.perception_module = PerceptionModule(self)
            self.reasoning_module = ReasoningModule(self)
            self.planning_module = PlanningModule(self)
            self.memory_system = MemorySystem(self)

            self.get_logger().info('Cognitive modules initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize cognitive modules: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def map_callback(self, msg):
        """Process occupancy grid map"""
        # Convert map to numpy array
        map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.current_map = map_data

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg

    def cognitive_processing_loop(self):
        """Main cognitive processing loop"""
        try:
            # Update world state from sensors
            world_update = self.perception_module.process_perception(
                self.current_image,
                self.current_scan,
                self.current_detections,
                self.current_map,
                self.current_robot_pose
            )

            # Merge with current world state
            self.current_state.objects = world_update.get('objects', [])
            self.current_state.environment_map = world_update.get('environment_map')
            self.current_state.recent_events.append({
                'type': 'perception_update',
                'data': world_update,
                'timestamp': time.time()
            })

            # Apply reasoning
            reasoning_result = self.reasoning_module.reason_about_world({
                'objects': self.current_state.objects,
                'scene_analysis': world_update.get('scene_analysis', {}),
                'robot_pose': self.current_state.robot_pose,
                'environment_map': self.current_state.environment_map,
                'goals': self.current_state.goals,
                'obstacles': self.current_state.obstacles
            })

            # Update cognitive state based on reasoning
            if reasoning_result['primary_intent'] == 'idle':
                self.cognitive_state = CognitiveState.IDLE
            elif reasoning_result['primary_intent'] in ['move_towards_goal', 'follow_person', 'avoid_obstacle']:
                self.cognitive_state = CognitiveState.PLANNING

            # Plan actions if in planning state
            if self.cognitive_state == CognitiveState.PLANNING:
                if self.current_goal:
                    plan = self.planning_module.plan_behavior(
                        {'type': 'navigate_to_location', 'pose': self.current_goal},
                        {
                            'robot_pose': self.current_state.robot_pose,
                            'environment_map': self.current_state.environment_map,
                            'objects': self.current_state.objects
                        }
                    )
                    self.current_plan = plan
                    self.plan_index = 0
                    self.cognitive_state = CognitiveState.EXECUTING

            # Execute plan if in execution state
            if self.cognitive_state == CognitiveState.EXECUTING:
                self.execute_plan_step()

            # Update status
            status_msg = String()
            status_msg.data = f'Cognitive State: {self.cognitive_state.value}, Reasoning: {reasoning_result["primary_intent"]}, Plan Steps: {len(self.current_plan)}'
            self.status_pub.publish(status_msg)

            self.get_logger().info(
                f'Cognitive State: {self.cognitive_state.value}, '
                f'Reasoning Intent: {reasoning_result["primary_intent"]}, '
                f'Objects: {len(self.current_state.objects)}, '
                f'Plan Length: {len(self.current_plan)}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in cognitive processing: {str(e)}')

    def execute_plan_step(self):
        """Execute current step in plan"""
        if not self.current_plan or self.plan_index >= len(self.current_plan):
            # Plan completed
            self.cognitive_state = CognitiveState.IDLE
            self.current_plan = []
            self.plan_index = 0
            return

        current_step = self.current_plan[self.plan_index]

        # Execute action based on plan step
        cmd = self.execute_action(current_step['action'], current_step['parameters'])

        if cmd is not None:
            self.cmd_vel_pub.publish(cmd)

        # Check if step is complete
        current_time = time.time()
        if current_time - self.last_action_time > current_step['duration']:
            # Move to next step
            self.plan_index += 1
            self.last_action_time = current_time

        # If plan is complete
        if self.plan_index >= len(self.current_plan):
            self.cognitive_state = CognitiveState.IDLE
            self.current_plan = []
            self.plan_index = 0

            self.get_logger().info('Plan execution completed')

    def execute_action(self, action_type: str, parameters: Dict) -> Twist:
        """Execute specific action type"""
        cmd = Twist()

        if action_type == 'move_forward':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif action_type == 'move_backward':
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
        elif action_type == 'turn_left':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif action_type == 'turn_right':
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif action_type == 'move_towards_object':
            # In a real implementation, this would navigate toward the object
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif action_type == 'avoid_object':
            # Simple obstacle avoidance
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3  # Turn away
        elif action_type == 'follow_object':
            # Simple following behavior
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif action_type == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Default: move forward slowly
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    cognitive_robot = CognitiveRobot()

    try:
        rclpy.spin(cognitive_robot)
    except KeyboardInterrupt:
        cognitive_robot.get_logger().info('Cognitive robot shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        cognitive_robot.cmd_vel_pub.publish(cmd)

        cognitive_robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Cognitive Planning with Neural Networks

Implementing more sophisticated cognitive planning using neural networks:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json

class AdvancedCognitivePlanner(Node):
    def __init__(self):
        super().__init__('advanced_cognitive_planner')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            Pose,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.plan_pub = self.create_publisher(
            String,
            '/advanced_plan',
            10
        )

        # Initialize advanced cognitive models
        self.cognitive_model = None
        self.initialize_advanced_cognitive_models()

        # State variables
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_map = None
        self.current_goal = None
        self.current_plan = []
        self.plan_index = 0

        # Memory and experience replay
        self.experience_buffer = []
        self.max_experience_size = 1000

        # Control timer
        self.advanced_planner_timer = self.create_timer(0.1, self.advanced_cognitive_loop)

        self.get_logger().info('Advanced cognitive planner initialized')

    def initialize_advanced_cognitive_models(self):
        """Initialize advanced cognitive models using neural networks"""
        try:
            # Cognitive planning neural network
            class CognitivePlannerNet(nn.Module):
                def __init__(self, input_dim=256, hidden_dim=512, output_dim=64):
                    super(CognitivePlannerNet, self).__init__()

                    # Input processing layers
                    self.vision_processor = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(),
                        nn.Linear(128 * 4 * 4, 256),
                        nn.ReLU()
                    )

                    # Sensor processing layers
                    self.sensor_processor = nn.Sequential(
                        nn.Linear(360, 256),  # Laser scan (assuming 360 points)
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU()
                    )

                    # Goal processing layers
                    self.goal_processor = nn.Sequential(
                        nn.Linear(3, 64),  # Goal pose (x, y, theta)
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU()
                    )

                    # Fusion layer
                    self.fusion = nn.Sequential(
                        nn.Linear(256 + 128 + 32, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    )

                    # Planning head
                    self.planning_head = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, output_dim)  # Action space
                    )

                    # Reasoning head
                    self.reasoning_head = nn.Sequential(
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16)  # Reasoning outputs
                    )

                def forward(self, vision_input, sensor_input, goal_input):
                    # Process vision input
                    vision_features = self.vision_processor(vision_input)

                    # Process sensor input
                    sensor_features = self.sensor_processor(sensor_input)

                    # Process goal input
                    goal_features = self.goal_processor(goal_input)

                    # Fuse all features
                    fused_features = torch.cat([vision_features, sensor_features, goal_features], dim=1)
                    fused_features = self.fusion(fused_features)

                    # Generate planning output
                    planning_output = self.planning_head(fused_features)

                    # Generate reasoning output
                    reasoning_output = self.reasoning_head(fused_features)

                    return planning_output, reasoning_output

            # Initialize model
            self.cognitive_model = CognitivePlannerNet()
            self.cognitive_model.eval()  # Set to evaluation mode

            self.get_logger().info('Advanced cognitive model initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize cognitive model: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.current_map = msg

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg

    def advanced_cognitive_loop(self):
        """Main advanced cognitive processing loop"""
        if (self.cognitive_model is None or
            self.current_image is None or
            self.current_scan is None):
            return

        try:
            # Prepare inputs for cognitive model
            vision_input = self.process_vision_input(self.current_image)
            sensor_input = self.process_sensor_input(self.current_scan)
            goal_input = self.process_goal_input(self.current_goal)

            if vision_input is None or sensor_input is None or goal_input is None:
                return

            # Run cognitive model
            with torch.no_grad():
                planning_output, reasoning_output = self.cognitive_model(
                    vision_input, sensor_input, goal_input
                )

            # Generate plan from model output
            plan = self.generate_plan_from_output(planning_output, reasoning_output)

            # Execute plan
            if plan:
                self.execute_advanced_plan(plan)

            # Store experience for learning
            self.store_experience(vision_input, sensor_input, goal_input, planning_output)

            self.get_logger().info(
                f'Advanced Cognitive - Plan length: {len(plan) if plan else 0}, '
                f'Reasoning confidence: {torch.max(reasoning_output).item():.3f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in advanced cognitive processing: {str(e)}')

    def process_vision_input(self, image_msg):
        """Process camera image for cognitive model"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Resize and normalize image
            image_resized = cv2.resize(cv_image, (64, 64))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
            image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
            image_tensor = torch.FloatTensor(image_tensor)

            return image_tensor

        except Exception as e:
            self.get_logger().error(f'Error processing vision input: {str(e)}')
            return None

    def process_sensor_input(self, scan_msg):
        """Process laser scan for cognitive model"""
        try:
            # Convert scan ranges to fixed-size tensor
            ranges = np.array(scan_msg.ranges)
            ranges = np.nan_to_num(ranges, nan=3.0)  # Replace NaN with max range
            ranges = np.clip(ranges, 0.0, 3.0)  # Clip to max range

            # Ensure we have exactly 360 points (pad or truncate)
            if len(ranges) < 360:
                # Pad with max range
                ranges = np.pad(ranges, (0, 360 - len(ranges)), mode='constant', constant_values=3.0)
            else:
                # Truncate to 360 points
                ranges = ranges[:360]

            # Normalize
            ranges_normalized = ranges / 3.0  # Normalize to [0, 1]

            scan_tensor = torch.FloatTensor(ranges_normalized).unsqueeze(0)  # Add batch dimension
            return scan_tensor

        except Exception as e:
            self.get_logger().error(f'Error processing sensor input: {str(e)}')
            return None

    def process_goal_input(self, goal_msg):
        """Process goal pose for cognitive model"""
        try:
            if goal_msg is None:
                # Default goal (origin)
                goal_tensor = torch.FloatTensor([[0.0, 0.0, 0.0]]).unsqueeze(0)  # Add batch dimension
                return goal_tensor

            # Extract position and orientation
            x = goal_msg.position.x
            y = goal_msg.position.y

            # Convert quaternion to euler angle (yaw)
            import math
            q = goal_msg.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny_cosp, cosy_cosp)

            goal_tensor = torch.FloatTensor([[x, y, theta]]).unsqueeze(0)  # Add batch dimension
            return goal_tensor

        except Exception as e:
            self.get_logger().error(f'Error processing goal input: {str(e)}')
            # Return default goal
            return torch.FloatTensor([[0.0, 0.0, 0.0]]).unsqueeze(0)

    def generate_plan_from_output(self, planning_output, reasoning_output):
        """Generate plan from model outputs"""
        try:
            # Convert model outputs to plan
            action_values = planning_output[0].numpy()  # Remove batch dimension

            # Interpret action values as movement commands
            # For simplicity, assume first two values are linear and angular velocities
            linear_vel = np.tanh(action_values[0]) * 0.5  # Scale to reasonable range
            angular_vel = np.tanh(action_values[1]) * 1.0

            # Create plan with single action
            plan = [{
                'action': 'move',
                'linear_x': float(linear_vel),
                'angular_z': float(angular_vel),
                'duration': 0.5  # 0.5 seconds per action
            }]

            return plan

        except Exception as e:
            self.get_logger().error(f'Error generating plan from output: {str(e)}')
            return None

    def execute_advanced_plan(self, plan):
        """Execute advanced cognitive plan"""
        if not plan or self.plan_index >= len(plan):
            return

        current_action = plan[self.plan_index]

        if current_action['action'] == 'move':
            cmd = Twist()
            cmd.linear.x = current_action['linear_x']
            cmd.angular.z = current_action['angular_z']

            # Limit velocities
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            self.cmd_vel_pub.publish(cmd)

            # Publish plan information
            plan_msg = String()
            plan_msg.data = json.dumps({
                'current_action': current_action,
                'plan_length': len(plan),
                'step_index': self.plan_index
            })
            self.plan_pub.publish(plan_msg)

            # Move to next action after duration
            if time.time() - self.last_action_time > current_action['duration']:
                self.plan_index += 1
                self.last_action_time = time.time()

                if self.plan_index >= len(plan):
                    self.plan_index = 0  # Reset for next planning cycle

    def store_experience(self, vision_input, sensor_input, goal_input, planning_output):
        """Store experience for learning and memory"""
        experience = {
            'vision': vision_input.detach().cpu().numpy() if vision_input is not None else None,
            'sensor': sensor_input.detach().cpu().numpy() if sensor_input is not None else None,
            'goal': goal_input.detach().cpu().numpy() if goal_input is not None else None,
            'action': planning_output.detach().cpu().numpy() if planning_output is not None else None,
            'timestamp': time.time()
        }

        self.experience_buffer.append(experience)

        # Keep only recent experiences
        if len(self.experience_buffer) > self.max_experience_size:
            self.experience_buffer = self.experience_buffer[-self.max_experience_size:]

def main(args=None):
    rclpy.init(args=args)
    advanced_planner = AdvancedCognitivePlanner()

    try:
        rclpy.spin(advanced_planner)
    except KeyboardInterrupt:
        advanced_planner.get_logger().info('Advanced cognitive planner shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_planner.cmd_vel_pub.publish(cmd)

        advanced_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Cognitive Reasoning with Knowledge Graphs

Implementing cognitive reasoning using knowledge graphs:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
import json

class KnowledgeGraphReasoner(Node):
    def __init__(self):
        super().__init__('knowledge_graph_reasoner')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            Pose,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.reasoning_pub = self.create_publisher(
            String,
            '/reasoning_output',
            10
        )

        # Initialize knowledge graph
        self.knowledge_graph = None
        self.initialize_knowledge_graph()

        # Initialize reasoning components
        self.reasoning_engine = None
        self.initialize_reasoning_engine()

        # State variables
        self.current_percepts = {}
        self.current_beliefs = {}
        self.reasoning_goals = []

        # Control timer
        self.reasoning_timer = self.create_timer(0.1, self.reasoning_loop)

        self.get_logger().info('Knowledge graph reasoner initialized')

    def initialize_knowledge_graph(self):
        """Initialize knowledge graph with robot domain knowledge"""
        try:
            # Create directed graph for knowledge representation
            self.knowledge_graph = nx.DiGraph()

            # Add object classes and their properties
            object_classes = [
                ('object', {'type': 'abstract'}),
                ('robot', {'type': 'object', 'movable': False}),
                ('human', {'type': 'object', 'movable': True, 'social': True}),
                ('furniture', {'type': 'object', 'movable': False}),
                ('table', {'type': 'furniture', 'surface': True}),
                ('chair', {'type': 'furniture', 'seatable': True}),
                ('box', {'type': 'object', 'movable': True, 'graspable': True}),
                ('ball', {'type': 'object', 'movable': True, 'graspable': True, 'rollable': True}),
                ('door', {'type': 'object', 'openable': True, 'passable_when_open': True}),
                ('wall', {'type': 'object', 'movable': False, 'obstacle': True}),
                ('floor', {'type': 'object', 'movable': False, 'walkable': True})
            ]

            for obj_class, attrs in object_classes:
                self.knowledge_graph.add_node(obj_class, **attrs)

            # Add spatial relationships
            spatial_relations = [
                ('left_of', {'symmetric': False, 'transitive': False}),
                ('right_of', {'symmetric': False, 'transitive': False}),
                ('in_front_of', {'symmetric': False, 'transitive': False}),
                ('behind', {'symmetric': False, 'transitive': False}),
                ('on_top_of', {'symmetric': False, 'transitive': True}),
                ('under', {'symmetric': False, 'transitive': True}),
                ('next_to', {'symmetric': True, 'transitive': False}),
                ('inside', {'symmetric': False, 'transitive': True}),
                ('outside', {'symmetric': False, 'transitive': False})
            ]

            for rel, attrs in spatial_relations:
                self.knowledge_graph.add_node(rel, type='relation', **attrs)

            # Add functional relationships
            functional_relations = [
                ('has_part', {'type': 'functional', 'transitive': True}),
                ('is_part_of', {'type': 'functional', 'transitive': True}),
                ('is_a', {'type': 'functional', 'transitive': True}),
                ('can_be', {'type': 'functional', 'transitive': False}),
                ('used_for', {'type': 'functional', 'transitive': False}),
                ('located_in', {'type': 'functional', 'transitive': True})
            ]

            for rel, attrs in functional_relations:
                self.knowledge_graph.add_node(rel, **attrs)

            # Add instance relationships based on domain knowledge
            instance_relations = [
                ('robot', 'is_a', 'object'),
                ('human', 'is_a', 'object'),
                ('table', 'is_a', 'furniture'),
                ('chair', 'is_a', 'furniture'),
                ('box', 'is_a', 'object'),
                ('ball', 'is_a', 'object'),
                ('door', 'is_a', 'object'),
                ('wall', 'is_a', 'object'),
                ('floor', 'is_a', 'object'),
                ('box', 'can_be', 'grasped'),
                ('ball', 'can_be', 'grasped'),
                ('table', 'used_for', 'placing_objects'),
                ('chair', 'used_for', 'sitting'),
                ('door', 'used_for', 'passage'),
                ('robot', 'used_for', 'assistance')
            ]

            for subj, pred, obj in instance_relations:
                self.knowledge_graph.add_edge(subj, obj, relation=pred)

            self.get_logger().info('Knowledge graph initialized with domain knowledge')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize knowledge graph: {str(e)}')

    def initialize_reasoning_engine(self):
        """Initialize reasoning engine components"""
        try:
            class ReasoningEngine:
                def __init__(self, knowledge_graph, node):
                    self.kg = knowledge_graph
                    self.node = node

                def perform_reasoning(self, percepts: Dict, goals: List[Dict]) -> Dict:
                    """Perform reasoning based on percepts and goals"""
                    results = {
                        'beliefs': {},
                        'inferences': [],
                        'action_recommendations': [],
                        'confidence_scores': {}
                    }

                    # Update beliefs based on percepts
                    beliefs = self.update_beliefs(percepts)
                    results['beliefs'] = beliefs

                    # Perform inference to derive new knowledge
                    inferences = self.perform_inference(percepts, beliefs)
                    results['inferences'] = inferences

                    # Generate action recommendations based on goals
                    action_recs = self.generate_action_recommendations(goals, beliefs, inferences)
                    results['action_recommendations'] = action_recs

                    # Calculate confidence scores
                    confidence_scores = self.calculate_confidence_scores(
                        percepts, beliefs, inferences, action_recs
                    )
                    results['confidence_scores'] = confidence_scores

                    return results

                def update_beliefs(self, percepts: Dict) -> Dict:
                    """Update beliefs based on current percepts"""
                    beliefs = {}

                    # Process object detections
                    if 'objects' in percepts:
                        for obj in percepts['objects']:
                            obj_class = obj.get('class', 'unknown')
                            obj_id = obj.get('id', 'unknown')
                            position = obj.get('position', {})

                            belief_key = f"object_at_{position.get('x', 0)}_{position.get('y', 0)}"
                            beliefs[belief_key] = {
                                'class': obj_class,
                                'confidence': obj.get('confidence', 0.0),
                                'position': position,
                                'timestamp': time.time()
                            }

                    # Process spatial relationships
                    if 'spatial_relationships' in percepts:
                        for rel in percepts['spatial_relationships']:
                            rel_key = f"{rel['object_id']}_is_{rel['relation']}_{rel['reference_object']}"
                            beliefs[rel_key] = {
                                'relationship': rel['relation'],
                                'confidence': rel.get('confidence', 0.7),
                                'timestamp': time.time()
                            }

                    return beliefs

                def perform_inference(self, percepts: Dict, beliefs: Dict) -> List[Dict]:
                    """Perform logical inference to derive new knowledge"""
                    inferences = []

                    # Infer object affordances based on class
                    for belief_key, belief in beliefs.items():
                        if belief.get('class'):
                            obj_class = belief['class']

                            # Check if object class has known affordances in knowledge graph
                            try:
                                for neighbor in self.kg.neighbors(obj_class):
                                    edge_data = self.kg.get_edge_data(obj_class, neighbor)
                                    if edge_data and edge_data.get('relation') == 'can_be':
                                        inference = {
                                            'type': 'affordance_inference',
                                            'subject': belief_key,
                                            'action': neighbor,
                                            'confidence': 0.8
                                        }
                                        inferences.append(inference)
                            except nx.NetworkXError:
                                # Node not in graph
                                continue

                    # Infer spatial relationships transitively
                    for belief_key, belief in beliefs.items():
                        if belief.get('relationship'):
                            rel = belief['relationship']
                            # Check if relationship is transitive in knowledge graph
                            try:
                                rel_node = self.kg.nodes[rel]
                                if rel_node.get('transitive', False):
                                    # Perform transitive closure inference
                                    # This would be more complex in a real implementation
                                    pass
                            except KeyError:
                                # Relationship not in graph
                                continue

                    return inferences

                def generate_action_recommendations(self, goals: List[Dict], beliefs: Dict, inferences: List[Dict]) -> List[Dict]:
                    """Generate action recommendations based on goals and current state"""
                    recommendations = []

                    for goal in goals:
                        if goal.get('type') == 'approach_object':
                            # Find object in beliefs
                            target_obj = None
                            for belief_key, belief in beliefs.items():
                                if belief.get('class') == goal.get('object_class'):
                                    target_obj = belief
                                    break

                            if target_obj:
                                recommendation = {
                                    'action': 'navigate_to_object',
                                    'parameters': {
                                        'position': target_obj['position'],
                                        'object_class': target_obj['class']
                                    },
                                    'priority': 10
                                }
                                recommendations.append(recommendation)

                        elif goal.get('type') == 'avoid_hazard':
                            # Find hazardous objects
                            for belief_key, belief in beliefs.items():
                                if belief.get('class') in ['fire', 'hole', 'dangerous_object']:
                                    recommendation = {
                                        'action': 'avoid_object',
                                        'parameters': {
                                            'position': belief['position'],
                                            'object_class': belief['class']
                                        },
                                        'priority': 15
                                    }
                                    recommendations.append(recommendation)

                        elif goal.get('type') == 'assist_human':
                            # Find humans in environment
                            for belief_key, belief in beliefs.items():
                                if belief.get('class') == 'human':
                                    recommendation = {
                                        'action': 'approach_human',
                                        'parameters': {
                                            'position': belief['position']
                                        },
                                        'priority': 8
                                    }
                                    recommendations.append(recommendation)

                    # Sort by priority
                    recommendations.sort(key=lambda x: x['priority'], reverse=True)

                    return recommendations

                def calculate_confidence_scores(self, percepts: Dict, beliefs: Dict, inferences: List[Dict], recommendations: List[Dict]) -> Dict:
                    """Calculate confidence scores for reasoning results"""
                    scores = {}

                    # Overall perceptual confidence
                    if 'objects' in percepts:
                        object_confidences = [obj.get('confidence', 0.0) for obj in percepts.get('objects', [])]
                        if object_confidences:
                            scores['perceptual_confidence'] = sum(object_confidences) / len(object_confidences)
                        else:
                            scores['perceptual_confidence'] = 0.0
                    else:
                        scores['perceptual_confidence'] = 0.0

                    # Inference confidence
                    if inferences:
                        inference_confidences = [inf.get('confidence', 0.5) for inf in inferences]
                        scores['inference_confidence'] = sum(inference_confidences) / len(inference_confidences)
                    else:
                        scores['inference_confidence'] = 0.5

                    # Action recommendation confidence
                    if recommendations:
                        rec_confidences = []
                        for rec in recommendations:
                            # Base confidence on perceptual and inference confidences
                            rec_conf = (scores['perceptual_confidence'] + scores['inference_confidence']) / 2.0
                            rec_confidences.append(rec_conf)

                        scores['action_confidence'] = sum(rec_confidences) / len(rec_confidences)
                    else:
                        scores['action_confidence'] = 0.5

                    # Overall reasoning confidence
                    all_scores = [v for v in scores.values()]
                    scores['overall_confidence'] = sum(all_scores) / len(all_scores) if all_scores else 0.5

                    return scores

            self.reasoning_engine = ReasoningEngine(self.knowledge_graph, self)
            self.get_logger().info('Reasoning engine initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize reasoning engine: {str(e)}')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        # Convert detections to internal format
        objects = []
        for detection in msg.detections:
            obj = {
                'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                'position': {
                    'x': detection.bbox.center.x,
                    'y': detection.bbox.center.y
                },
                'id': id(detection)  # Simple ID based on object identity
            }
            objects.append(obj)

        self.current_percepts['objects'] = objects

    def goal_callback(self, msg):
        """Process navigation goal"""
        if self.current_goal is None:
            self.current_goals = []

        # Add new goal to list
        goal_info = {
            'type': 'navigate_to_location',
            'position': {
                'x': msg.position.x,
                'y': msg.position.y
            },
            'priority': 5
        }
        self.current_goals.append(goal_info)

    def reasoning_loop(self):
        """Main reasoning loop"""
        if (self.reasoning_engine is None or
            not self.current_percepts):
            return

        try:
            # Perform reasoning
            reasoning_results = self.reasoning_engine.perform_reasoning(
                self.current_percepts,
                self.current_goals
            )

            # Generate actions based on recommendations
            if reasoning_results['action_recommendations']:
                primary_action = reasoning_results['action_recommendations'][0]

                cmd = self.convert_action_to_command(primary_action)
                if cmd is not None:
                    self.cmd_vel_pub.publish(cmd)

            # Publish reasoning results
            reasoning_msg = String()
            reasoning_msg.data = json.dumps({
                'beliefs': reasoning_results['beliefs'],
                'inferences': reasoning_results['inferences'],
                'recommendations': reasoning_results['action_recommendations'],
                'confidences': reasoning_results['confidence_scores']
            })
            self.reasoning_pub.publish(reasoning_msg)

            self.get_logger().info(
                f'Reasoning - Objects: {len(self.current_percepts.get("objects", []))}, '
                f'Recommendations: {len(reasoning_results["action_recommendations"])}, '
                f'Overall Confidence: {reasoning_results["confidence_scores"].get("overall_confidence", 0):.3f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in reasoning loop: {str(e)}')

    def convert_action_to_command(self, action_rec: Dict) -> Twist:
        """Convert action recommendation to robot command"""
        cmd = Twist()

        action_type = action_rec['action']
        params = action_rec['parameters']

        if action_type == 'navigate_to_object':
            # Simple navigation toward object
            pos = params['position']
            # In a real implementation, this would use path planning
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif action_type == 'avoid_object':
            # Simple avoidance behavior
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
        elif action_type == 'approach_human':
            # Approach human behavior
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
        elif action_type == 'navigate_to_pose':
            # Navigate to specific pose
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Default: move forward slowly
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    reasoner = KnowledgeGraphReasoner()

    try:
        rclpy.spin(reasoner)
    except KeyboardInterrupt:
        reasoner.get_logger().info('Knowledge graph reasoner shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        reasoner.cmd_vel_pub.publish(cmd)

        reasoner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Planning with Uncertainty and Probabilistic Reasoning

Implementing planning systems that handle uncertainty:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
from typing import Dict, List, Tuple
import json
from scipy.stats import norm

class UncertainPlanningNode(Node):
    def __init__(self):
        super().__init__('uncertain_planning')

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            Pose,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.uncertainty_pub = self.create_publisher(
            Float32,
            '/planning_uncertainty',
            10
        )

        self.belief_pub = self.create_publisher(
            String,
            '/robot_belief_state',
            10
        )

        # Initialize probabilistic planning components
        self.belief_state = None
        self.uncertainty_estimator = None
        self.probabilistic_planner = None
        self.initialize_probabilistic_components()

        # State variables
        self.current_scan = None
        self.current_detections = None
        self.current_map = None
        self.current_goal = None
        self.current_plan = []
        self.plan_index = 0

        # Belief state variables
        self.position_belief = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'covariance': np.eye(3)}
        self.object_beliefs = {}  # Dictionary of object location probabilities

        # Control timer
        self.uncertainty_timer = self.create_timer(0.1, self.uncertain_planning_loop)

        self.get_logger().info('Uncertain planning node initialized')

    def initialize_probabilistic_components(self):
        """Initialize probabilistic planning components"""
        try:
            # Uncertainty estimator
            class UncertaintyEstimator:
                def __init__(self, node):
                    self.node = node

                def estimate_sensor_uncertainty(self, scan_msg, detection_msg):
                    """Estimate uncertainty from sensor data"""
                    uncertainty_metrics = {
                        'perceptual_uncertainty': 0.0,
                        'localization_uncertainty': 0.0,
                        'mapping_uncertainty': 0.0,
                        'prediction_uncertainty': 0.0
                    }

                    # Estimate perceptual uncertainty from detection confidences
                    if detection_msg and detection_msg.detections:
                        confidences = [det.results[0].hypothesis.score if det.results else 0.0
                                     for det in detection_msg.detections]
                        if confidences:
                            uncertainty_metrics['perceptual_uncertainty'] = 1.0 - np.mean(confidences)

                    # Estimate mapping uncertainty from laser scan
                    if scan_msg and scan_msg.ranges:
                        valid_ranges = [r for r in scan_msg.ranges if np.isfinite(r)]
                        if valid_ranges:
                            # Higher variance in ranges indicates higher uncertainty
                            range_variance = np.var(valid_ranges)
                            uncertainty_metrics['mapping_uncertainty'] = min(1.0, range_variance / 10.0)

                    # Estimate localization uncertainty (simplified)
                    uncertainty_metrics['localization_uncertainty'] = 0.1  # Default

                    # Overall uncertainty
                    uncertainty_metrics['total_uncertainty'] = np.mean(list(uncertainty_metrics.values()))

                    return uncertainty_metrics

            # Probabilistic planner
            class ProbabilisticPlanner:
                def __init__(self, node):
                    self.node = node

                def plan_with_uncertainty(self, goal_pose, belief_state, uncertainty_metrics):
                    """Generate plan considering uncertainty"""
                    # In a real implementation, this would use probabilistic roadmaps,
                    # belief space planning, or other uncertainty-aware planning methods
                    # For this example, we'll implement a simplified approach

                    plan = []

                    # Calculate nominal path to goal
                    nominal_path = self.calculate_nominal_path(goal_pose, belief_state)

                    # Adjust path based on uncertainty
                    adjusted_path = self.adjust_path_for_uncertainty(
                        nominal_path, uncertainty_metrics
                    )

                    # Convert to action plan
                    action_plan = self.convert_path_to_actions(adjusted_path, uncertainty_metrics)

                    return action_plan

                def calculate_nominal_path(self, goal_pose, belief_state):
                    """Calculate nominal path to goal"""
                    # Simplified path calculation
                    current_pose = belief_state.get('position', {'x': 0.0, 'y': 0.0, 'theta': 0.0})

                    dx = goal_pose.position.x - current_pose['x']
                    dy = goal_pose.position.y - current_pose['y']
                    distance = math.sqrt(dx*dx + dy*dy)

                    path = []
                    if distance > 0.1:  # If goal is not too close
                        steps = max(10, int(distance * 10))  # 10 steps per meter

                        for i in range(steps + 1):
                            t = i / steps
                            intermediate_pose = Pose()
                            intermediate_pose.position.x = current_pose['x'] + t * dx
                            intermediate_pose.position.y = current_pose['y'] + t * dy
                            # For simplicity, keep current orientation
                            intermediate_pose.orientation.z = math.sin(current_pose['theta'] / 2)
                            intermediate_pose.orientation.w = math.cos(current_pose['theta'] / 2)

                            path.append(intermediate_pose)

                    return path

                def adjust_path_for_uncertainty(self, nominal_path, uncertainty_metrics):
                    """Adjust path based on uncertainty levels"""
                    adjusted_path = []

                    total_uncertainty = uncertainty_metrics.get('total_uncertainty', 0.0)

                    for i, pose in enumerate(nominal_path):
                        # Adjust pose based on uncertainty
                        adjusted_pose = Pose()
                        adjusted_pose.position = pose.position
                        adjusted_pose.orientation = pose.orientation

                        # In high uncertainty, move more cautiously
                        if total_uncertainty > 0.5:  # High uncertainty threshold
                            # Add safety margins and reduce step size
                            if i > 0:  # Not the first point
                                prev_pose = adjusted_path[-1]
                                # Reduce the step size in high uncertainty
                                dx = pose.position.x - prev_pose.position.x
                                dy = pose.position.y - prev_pose.position.y
                                adjusted_pose.position.x = prev_pose.position.x + dx * 0.8
                                adjusted_pose.position.y = prev_pose.position.y + dy * 0.8

                        adjusted_path.append(adjusted_pose)

                    return adjusted_path

                def convert_path_to_actions(self, path, uncertainty_metrics):
                    """Convert path to action sequence"""
                    actions = []

                    for i in range(len(path) - 1):
                        current_pose = path[i]
                        next_pose = path[next_pose]

                        # Calculate required movement
                        dx = next_pose.position.x - current_pose.position.x
                        dy = next_pose.position.y - current_pose.position.y
                        dist = math.sqrt(dx*dx + dy*dy)

                        # Calculate orientation change
                        current_theta = 2 * math.atan2(current_pose.orientation.z, current_pose.orientation.w)
                        next_theta = 2 * math.atan2(next_pose.orientation.z, next_pose.orientation.w)
                        dtheta = next_theta - current_theta

                        # Create action
                        action = {
                            'type': 'move_to_pose',
                            'linear_distance': dist,
                            'angular_change': dtheta,
                            'duration': dist / 0.3 + abs(dtheta) / 0.5,  # Estimated duration
                            'uncertainty_factor': uncertainty_metrics.get('total_uncertainty', 0.0)
                        }

                        actions.append(action)

                    return actions

            self.uncertainty_estimator = UncertaintyEstimator(self)
            self.probabilistic_planner = ProbabilisticPlanner(self)

            self.get_logger().info('Probabilistic planning components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize probabilistic components: {str(e)}')

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def map_callback(self, msg):
        """Process occupancy grid"""
        self.current_map = msg

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg

    def uncertain_planning_loop(self):
        """Main uncertain planning loop"""
        if (self.uncertainty_estimator is None or
            self.probabilistic_planner is None or
            self.current_scan is None or
            self.current_goal is None):
            return

        try:
            # Estimate uncertainty
            uncertainty_metrics = self.uncertainty_estimator.estimate_sensor_uncertainty(
                self.current_scan, self.current_detections
            )

            # Update belief state based on sensor data
            self.update_belief_state(uncertainty_metrics)

            # Plan with uncertainty consideration
            plan = self.probabilistic_planner.plan_with_uncertainty(
                self.current_goal, self.belief_state, uncertainty_metrics
            )

            # Execute plan considering uncertainty
            if plan:
                self.execute_uncertain_plan(plan, uncertainty_metrics)

            # Publish uncertainty metrics
            uncertainty_msg = Float32()
            uncertainty_msg.data = uncertainty_metrics.get('total_uncertainty', 0.0)
            self.uncertainty_pub.publish(uncertainty_msg)

            # Publish belief state
            belief_msg = String()
            belief_msg.data = json.dumps({
                'position': self.position_belief,
                'uncertainty_metrics': uncertainty_metrics,
                'plan_length': len(plan) if plan else 0
            })
            self.belief_pub.publish(belief_msg)

            self.get_logger().info(
                f'Uncertain Planning - Total Uncertainty: {uncertainty_metrics.get("total_uncertainty", 0):.3f}, '
                f'Plan Length: {len(plan) if plan else 0}, '
                f'Localization Uncertainty: {uncertainty_metrics.get("localization_uncertainty", 0):.3f}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in uncertain planning: {str(e)}')

    def update_belief_state(self, uncertainty_metrics):
        """Update belief state based on sensor data and uncertainty"""
        # In a real implementation, this would use Kalman filters, particle filters, etc.
        # For this example, we'll implement a simplified belief update

        # Update position belief with some uncertainty
        if self.current_scan:
            # Use laser scan to refine position estimate
            ranges = np.array(self.current_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]

            if len(valid_ranges) > 0:
                # Simplified position refinement based on scan
                mean_range = np.mean(valid_ranges)
                # Adjust uncertainty based on scan quality
                scan_quality = len(valid_ranges) / len(ranges)  # Ratio of valid ranges
                self.position_belief['covariance'][0, 0] *= (2.0 - scan_quality)  # Increase confidence with better scans

        # Update object beliefs
        if self.current_detections:
            for detection in self.current_detections.detections:
                obj_class = detection.results[0].hypothesis.class_id if detection.results else 'unknown'
                confidence = detection.results[0].hypothesis.score if detection.results else 0.0

                obj_id = f"{obj_class}_{id(detection)}"  # Simple object ID

                # Update object location belief
                if obj_id not in self.object_beliefs:
                    self.object_beliefs[obj_id] = {
                        'position': {
                            'x': detection.bbox.center.x,
                            'y': detection.bbox.center.y
                        },
                        'confidence': confidence,
                        'covariance': np.eye(2) * (1.0 - confidence)  # Higher uncertainty for lower confidence
                    }
                else:
                    # Update existing belief with new observation
                    prev_pos = self.object_beliefs[obj_id]['position']
                    new_pos = {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y
                    }

                    # Simple weighted update based on confidence
                    weight = confidence
                    updated_pos = {
                        'x': prev_pos['x'] * (1-weight) + new_pos['x'] * weight,
                        'y': prev_pos['y'] * (1-weight) + new_pos['y'] * weight
                    }

                    self.object_beliefs[obj_id]['position'] = updated_pos
                    self.object_beliefs[obj_id]['confidence'] = max(
                        self.object_beliefs[obj_id]['confidence'],
                        confidence
                    )

        # Create belief state dictionary
        self.belief_state = {
            'position': self.position_belief,
            'objects': self.object_beliefs,
            'uncertainty_metrics': uncertainty_metrics
        }

    def execute_uncertain_plan(self, plan, uncertainty_metrics):
        """Execute plan considering uncertainty"""
        if not plan or self.plan_index >= len(plan):
            return

        current_action = plan[self.plan_index]
        uncertainty_factor = current_action.get('uncertainty_factor', 0.0)

        # Adjust action execution based on uncertainty
        cmd = Twist()

        if current_action['type'] == 'move_to_pose':
            # In high uncertainty, move more slowly and cautiously
            base_linear_speed = 0.3
            base_angular_speed = 0.5

            # Reduce speed based on uncertainty
            speed_factor = max(0.1, 1.0 - uncertainty_factor)
            cmd.linear.x = base_linear_speed * speed_factor
            cmd.angular.z = current_action['angular_change'] * speed_factor

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        self.cmd_vel_pub.publish(cmd)

        # Move to next action after appropriate time considering uncertainty
        base_duration = current_action.get('duration', 1.0)
        uncertainty_duration_factor = 1.0 + uncertainty_factor  # Spend more time in uncertain situations
        adjusted_duration = base_duration * uncertainty_duration_factor

        if time.time() - self.last_action_time > adjusted_duration:
            self.plan_index += 1
            self.last_action_time = time.time()

            if self.plan_index >= len(plan):
                self.plan_index = 0  # Reset for next planning cycle

def main(args=None):
    rclpy.init(args=args)
    uncertain_planner = UncertainPlanningNode()

    try:
        rclpy.spin(uncertain_planner)
    except KeyboardInterrupt:
        uncertain_planner.get_logger().info('Uncertain planning node shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        uncertain_planner.cmd_vel_pub.publish(cmd)

        uncertain_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Learning and Adaptation in Cognitive Systems

Implementing learning and adaptation mechanisms:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
from typing import Dict, List, Tuple

class CognitiveLearningNode(Node):
    def __init__(self):
        super().__init__('cognitive_learning')

        # Publishers and subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.feedback_sub = self.create_subscription(
            String,
            '/task_feedback',
            self.feedback_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.performance_pub = self.create_publisher(
            Float32,
            '/learning_performance',
            10
        )

        self.adaptation_pub = self.create_publisher(
            String,
            '/adaptation_strategy',
            10
        )

        # Initialize learning components
        self.learning_model = None
        self.experience_buffer = None
        self.performance_evaluator = None
        self.initialize_learning_components()

        # State variables
        self.current_odom = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.current_feedback = None

        # Learning state
        self.episode_count = 0
        self.step_count = 0
        self.performance_history = deque(maxlen=100)
        self.learning_enabled = True

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)  # Store experiences for learning

        # Control timer
        self.learning_timer = self.create_timer(0.1, self.learning_loop)

        self.get_logger().info('Cognitive learning node initialized')

    def initialize_learning_components(self):
        """Initialize learning and adaptation components"""
        try:
            # Learning model (simplified neural network)
            class LearningModel(nn.Module):
                def __init__(self, input_size=360, hidden_size=256, output_size=2):
                    super(LearningModel, self).__init__()

                    # Input processing (laser scan)
                    self.input_layer = nn.Linear(input_size, hidden_size)

                    # Hidden layers
                    self.hidden1 = nn.Linear(hidden_size, hidden_size)
                    self.hidden2 = nn.Linear(hidden_size, hidden_size)

                    # Output layers for action prediction
                    self.action_layer = nn.Linear(hidden_size, output_size)

                    # Dropout for regularization
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    x = torch.relu(self.input_layer(x))
                    x = self.dropout(x)
                    x = torch.relu(self.hidden1(x))
                    x = self.dropout(x)
                    x = torch.relu(self.hidden2(x))
                    x = self.dropout(x)
                    x = self.action_layer(x)
                    return x

            # Performance evaluator
            class PerformanceEvaluator:
                def __init__(self, node):
                    self.node = node

                def evaluate_performance(self, state, action, next_state, reward, done):
                    """Evaluate performance based on state transitions and rewards"""
                    performance_metrics = {
                        'efficiency': 0.0,
                        'safety': 0.0,
                        'goal_achievement': 0.0,
                        'smoothness': 0.0
                    }

                    # Calculate efficiency (how direct the path to goal was)
                    if hasattr(self.node, 'current_goal') and self.node.current_goal:
                        goal_pos = self.node.current_goal.position
                        if next_state.get('position'):
                            current_pos = next_state['position']
                            dist_to_goal = np.sqrt(
                                (goal_pos.x - current_pos['x'])**2 +
                                (goal_pos.y - current_pos['y'])**2
                            )
                            performance_metrics['efficiency'] = 1.0 / (1.0 + dist_to_goal)

                    # Calculate safety (based on proximity to obstacles)
                    if 'scan_ranges' in next_state:
                        scan_ranges = next_state['scan_ranges']
                        valid_ranges = [r for r in scan_ranges if np.isfinite(r)]
                        if valid_ranges:
                            min_range = min(valid_ranges)
                            # Higher safety score for larger minimum range to obstacles
                            performance_metrics['safety'] = min(1.0, min_range / 1.0)

                    # Calculate goal achievement (simplified)
                    if done and reward > 0:
                        performance_metrics['goal_achievement'] = 1.0
                    elif done and reward < 0:
                        performance_metrics['goal_achievement'] = 0.0
                    else:
                        performance_metrics['goal_achievement'] = 0.5  # In progress

                    # Calculate smoothness (based on action consistency)
                    if hasattr(self.node, 'previous_action') and self.node.previous_action:
                        current_action = action
                        action_difference = np.abs(
                            current_action[0] - self.node.previous_action[0]
                        ) + np.abs(current_action[1] - self.node.previous_action[1])
                        performance_metrics['smoothness'] = max(0.0, 1.0 - action_difference)

                    return performance_metrics

            # Initialize components
            self.learning_model = LearningModel()
            self.performance_evaluator = PerformanceEvaluator(self)

            # Set up optimizer
            self.optimizer = optim.Adam(self.learning_model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()

            self.get_logger().info('Learning components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize learning components: {str(e)}')

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_odom = msg

    def scan_callback(self, msg):
        """Process laser scan"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.current_command = msg

    def feedback_callback(self, msg):
        """Process task feedback"""
        self.current_feedback = msg

    def learning_loop(self):
        """Main learning and adaptation loop"""
        if (self.learning_model is None or
            self.current_scan is None or
            self.current_odom is None):
            return

        try:
            # Prepare state representation
            state = self.prepare_state()
            if state is None:
                return

            # Get action from model
            action = self.get_action(state)

            # Execute action
            cmd = self.convert_action_to_command(action)
            if cmd is not None:
                self.cmd_vel_pub.publish(cmd)

            # Store experience for learning
            if hasattr(self, 'previous_state') and self.previous_state is not None:
                experience = {
                    'state': self.previous_state,
                    'action': self.previous_action,
                    'next_state': state,
                    'reward': self.calculate_reward(state),
                    'done': self.is_episode_done(state)
                }
                self.experience_buffer.append(experience)

            # Update previous state and action
            self.previous_state = state
            self.previous_action = action

            # Perform learning from experience replay
            if len(self.experience_buffer) > 100 and self.learning_enabled:
                self.perform_learning_step()

            # Evaluate performance
            if hasattr(self, 'previous_state') and self.previous_state is not None:
                performance = self.performance_evaluator.evaluate_performance(
                    self.previous_state, action, state,
                    self.calculate_reward(state), self.is_episode_done(state)
                )

                # Store performance history
                avg_performance = np.mean(list(performance.values()))
                self.performance_history.append(avg_performance)

                # Publish performance
                perf_msg = Float32()
                perf_msg.data = avg_performance
                self.performance_pub.publish(perf_msg)

                # Check for adaptation needs
                self.check_adaptation_needs(performance)

            self.step_count += 1

            if self.step_count % 100 == 0:  # Log every 100 steps
                self.get_logger().info(
                    f'Learning - Episode: {self.episode_count}, '
                    f'Step: {self.step_count}, '
                    f'Buffer Size: {len(self.experience_buffer)}, '
                    f'Avg Performance: {avg_performance:.3f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in learning loop: {str(e)}')

    def prepare_state(self) -> Dict:
        """Prepare state representation from sensor data"""
        if self.current_scan is None or self.current_odom is None:
            return None

        # Process laser scan
        scan_ranges = np.array(self.current_scan.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, nan=3.0)  # Replace NaN with max range
        scan_ranges = np.clip(scan_ranges, 0.0, 3.0)  # Clip to max range

        # Process odometry
        position = {
            'x': self.current_odom.pose.pose.position.x,
            'y': self.current_odom.pose.pose.position.y
        }

        # Convert quaternion to euler angle (yaw)
        import math
        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        # Process detections
        detected_objects = []
        if self.current_detections:
            for detection in self.current_detections.detections:
                obj_info = {
                    'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                    'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                    'position': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y
                    }
                }
                detected_objects.append(obj_info)

        state = {
            'scan_ranges': scan_ranges.tolist(),
            'position': position,
            'orientation': theta,
            'linear_velocity': self.current_odom.twist.twist.linear.x,
            'angular_velocity': self.current_odom.twist.twist.angular.z,
            'detected_objects': detected_objects
        }

        return state

    def get_action(self, state: Dict) -> List[float]:
        """Get action from learning model"""
        try:
            # Prepare input tensor from state
            scan_tensor = torch.FloatTensor(state['scan_ranges']).unsqueeze(0)  # Add batch dimension

            # Run model
            with torch.no_grad():
                action_output = self.learning_model(scan_tensor)

            # Convert to action values
            action_values = action_output[0].numpy()

            # Clamp to reasonable ranges
            linear_vel = np.clip(action_values[0], -1.0, 1.0)
            angular_vel = np.clip(action_values[1], -1.0, 1.0)

            return [linear_vel, angular_vel]

        except Exception as e:
            self.get_logger().error(f'Error getting action: {str(e)}')
            # Return default action
            return [0.3, 0.0]  # Move forward slowly

    def convert_action_to_command(self, action: List[float]) -> Twist:
        """Convert action to robot command"""
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

    def calculate_reward(self, state: Dict) -> float:
        """Calculate reward based on current state"""
        reward = 0.0

        # Positive reward for forward movement (but not too fast)
        if 'linear_velocity' in state:
            linear_vel = state['linear_velocity']
            if 0.1 < linear_vel < 0.5:
                reward += 0.1  # Encourage forward movement
            elif linear_vel > 0.5:
                reward -= 0.05  # Penalize excessive speed

        # Penalty for being too close to obstacles
        if 'scan_ranges' in state:
            scan_ranges = state['scan_ranges']
            valid_ranges = [r for r in scan_ranges if r > 0 and r < float('inf')]
            if valid_ranges:
                min_range = min(valid_ranges)
                if min_range < 0.5:  # Too close to obstacle
                    reward -= (0.5 - min_range) * 10  # Higher penalty for closer obstacles

        # Positive reward for detecting interesting objects
        if 'detected_objects' in state:
            for obj in state['detected_objects']:
                if obj['confidence'] > 0.7:  # High confidence detection
                    if obj['class'] in ['person', 'object']:
                        reward += 0.05  # Small positive reward for detecting objects

        return reward

    def is_episode_done(self, state: Dict) -> bool:
        """Check if episode is done"""
        # For this example, we'll use a simple criterion
        # In a real implementation, you'd have specific termination conditions

        # Check for collision (simplified)
        if 'scan_ranges' in state:
            scan_ranges = state['scan_ranges']
            valid_ranges = [r for r in scan_ranges if r > 0 and r < float('inf')]
            if valid_ranges:
                min_range = min(valid_ranges)
                if min_range < 0.2:  # Collision threshold
                    return True

        return False

    def perform_learning_step(self):
        """Perform a learning step using experience replay"""
        if len(self.experience_buffer) < 32:  # Need minimum batch size
            return

        try:
            # Sample random batch from experience buffer
            batch_size = min(32, len(self.experience_buffer))
            batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)

            states_batch = []
            actions_batch = []
            next_states_batch = []
            rewards_batch = []
            dones_batch = []

            for idx in batch_indices:
                exp = self.experience_buffer[idx]
                states_batch.append(exp['state']['scan_ranges'])
                actions_batch.append(exp['action'])
                next_states_batch.append(exp['next_state']['scan_ranges'])
                rewards_batch.append(exp['reward'])
                dones_batch.append(exp['done'])

            # Convert to tensors
            states_tensor = torch.FloatTensor(states_batch)
            actions_tensor = torch.FloatTensor(actions_batch)
            rewards_tensor = torch.FloatTensor(rewards_batch).unsqueeze(1)
            dones_tensor = torch.BoolTensor(dones_batch).unsqueeze(1)

            # Forward pass
            predicted_actions = self.learning_model(states_tensor)

            # Calculate loss (simplified - in DQN you'd use target network)
            loss = self.criterion(predicted_actions, actions_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log learning progress occasionally
            if self.step_count % 1000 == 0:
                self.get_logger().info(f'Learning - Loss: {loss.item():.6f}')

        except Exception as e:
            self.get_logger().error(f'Error in learning step: {str(e)}')

    def check_adaptation_needs(self, performance: Dict):
        """Check if adaptation is needed based on performance"""
        # Calculate average performance over recent episodes
        if len(self.performance_history) > 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            overall_avg = np.mean(self.performance_history)

            # If performance is declining, consider adaptation strategies
            if recent_avg < overall_avg * 0.8:  # 20% decline
                adaptation_strategy = {
                    'type': 'performance_degradation_detected',
                    'recent_avg': float(recent_avg),
                    'overall_avg': float(overall_avg),
                    'recommendation': 'increase_exploration'
                }

                strategy_msg = String()
                strategy_msg.data = json.dumps(adaptation_strategy)
                self.adaptation_pub.publish(strategy_msg)

                self.get_logger().warn(
                    f'Performance degradation detected: '
                    f'Recent: {recent_avg:.3f}, Overall: {overall_avg:.3f}'
                )

    def toggle_learning(self, enable: bool):
        """Toggle learning on/off"""
        self.learning_enabled = enable
        status = "enabled" if enable else "disabled"
        self.get_logger().info(f'Learning {status}')

def main(args=None):
    rclpy.init(args=args)
    learning_node = CognitiveLearningNode()

    try:
        rclpy.spin(learning_node)
    except KeyboardInterrupt:
        learning_node.get_logger().info('Cognitive learning node shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        learning_node.cmd_vel_pub.publish(cmd)

        learning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Cognitive Robotics

1. **Hierarchical Planning**: Implement hierarchical planning with multiple levels of abstraction
2. **Uncertainty Management**: Account for sensor noise, actuator errors, and environmental uncertainty
3. **Context Awareness**: Maintain awareness of environment, situation, and task context
4. **Memory Management**: Efficiently store and retrieve relevant information
5. **Learning Efficiency**: Balance exploration vs exploitation in learning
6. **Safety First**: Implement safety checks and fail-safe behaviors
7. **Real-time Performance**: Ensure cognitive processes meet real-time requirements
8. **Modularity**: Design components to be modular and reusable

### Physical Grounding and Simulation-to-Real Mapping

When implementing cognitive robotics systems:

- **Embodied Cognition**: Ensure all cognitive processes are grounded in physical reality
- **Sensorimotor Coupling**: Maintain tight coupling between perception, cognition, and action
- **Reality Mapping**: Ensure cognitive models accurately reflect real-world physics and constraints
- **Latency Considerations**: Account for processing delays in cognitive systems
- **Resource Constraints**: Consider computational and memory limitations on real hardware
- **Safety Systems**: Implement robust safety mechanisms around cognitive decision-making
- **Validation**: Validate cognitive behaviors in both simulation and real environments

### Troubleshooting Cognitive Robotics Issues

Common cognitive robotics problems and solutions:

- **Planning Failures**: Check environment representation and goal formulation
- **Learning Inefficiency**: Adjust learning rates and exploration strategies
- **Uncertainty Accumulation**: Implement belief state maintenance and correction
- **Real-time Violations**: Optimize algorithms and consider parallel processing
- **Context Loss**: Implement better context tracking and recovery mechanisms
- **Integration Issues**: Ensure proper data flow between cognitive components

### Summary

This chapter covered cognitive robotics and planning systems that enable robots to think, reason, and plan complex behaviors. You learned about implementing cognitive architectures with perception, reasoning, and planning components, using neural networks for advanced cognitive processing, implementing knowledge graphs for reasoning, handling uncertainty in planning, and incorporating learning and adaptation mechanisms. Cognitive robotics systems enable robots to perform complex autonomous behaviors by combining perception, reasoning, and action in an integrated framework. The integration of learning and adaptation allows robots to improve their performance over time and adapt to changing environments. In the next chapter, we'll explore multimodal fusion for integrated perception and action systems.