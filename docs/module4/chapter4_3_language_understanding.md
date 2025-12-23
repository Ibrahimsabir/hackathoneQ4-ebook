# Module 4: Vision–Language–Action (VLA)

## Chapter 4.3: Language Understanding and Processing

This chapter focuses on natural language understanding and processing for robotics applications. Language understanding enables robots to interpret human commands, engage in conversations, and follow complex instructions expressed in natural language.

### Understanding Language Processing in Robotics

Natural language processing (NLP) in robotics involves converting human language into actionable commands that robots can execute. Key aspects include:

- **Speech Recognition**: Converting spoken language to text
- **Intent Recognition**: Understanding the purpose behind commands
- **Entity Extraction**: Identifying relevant objects, locations, and parameters
- **Command Generation**: Converting natural language to robot actions
- **Dialogue Management**: Handling multi-turn conversations
- **Context Awareness**: Understanding contextual references and pronouns

### Language Processing Architecture

The language processing pipeline typically follows this architecture:

```
+-------------------+
|   Spoken Input    |
|   (Microphone)    |
+-------------------+
|   Speech-to-Text  |
|   (ASR)           |
+-------------------+
|   Text Processing |
|   (NLU)           |
+-------------------+
|   Intent & Entity |
|   Extraction      |
+-------------------+
|   Command         |
|   Generation      |
+-------------------+
|   Action Mapping  |
|   (Robot Actions) |
+-------------------+
```

### Basic Language Processing Node

Implementing a foundational language processing node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
import numpy as np
import re
import spacy
from typing import Dict, List, Tuple

class LanguageProcessor(Node):
    def __init__(self):
        super().__init__('language_processor')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.response_pub = self.create_publisher(
            String,
            '/robot_response',
            10
        )

        # Initialize language processing components
        self.nlp_model = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.initialize_language_components()

        # State variables
        self.current_command = None
        self.current_detections = None
        self.context = {}  # Store conversation context

        # Intent-action mapping
        self.intent_actions = {
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'approach_object': self.approach_object,
            'avoid_object': self.avoid_object,
            'stop': self.stop_robot,
            'follow_object': self.follow_object,
            'pick_up': self.pick_up_object,
            'put_down': self.put_down_object
        }

        # Control timer
        self.lang_timer = self.create_timer(0.1, self.language_processing_loop)

        self.get_logger().info('Language processor initialized')

    def initialize_language_components(self):
        """Initialize language processing components"""
        try:
            # For this example, we'll use a simple rule-based approach
            # In a real implementation, you'd use spaCy, transformers, etc.

            # Intent classifier (simplified)
            self.intent_classifier = {
                'move_forward': [
                    'move forward', 'go forward', 'move ahead', 'go ahead',
                    'forward', 'straight', 'ahead', 'go straight'
                ],
                'move_backward': [
                    'move backward', 'go backward', 'move back', 'go back',
                    'backward', 'back', 'reverse'
                ],
                'turn_left': [
                    'turn left', 'go left', 'turn left', 'left'
                ],
                'turn_right': [
                    'turn right', 'go right', 'turn right', 'right'
                ],
                'approach_object': [
                    'go to', 'move to', 'approach', 'come to', 'go near',
                    'approach the', 'move toward', 'go toward'
                ],
                'avoid_object': [
                    'avoid', 'stay away', 'move away', 'go around', 'steer clear'
                ],
                'stop': [
                    'stop', 'halt', 'pause', 'freeze', 'cease', 'quit'
                ],
                'follow_object': [
                    'follow', 'chase', 'track', 'go after', 'follow the'
                ],
                'pick_up': [
                    'pick up', 'grab', 'take', 'lift', 'collect', 'get'
                ],
                'put_down': [
                    'put down', 'drop', 'place', 'set down', 'release'
                ]
            }

            # Entity extractor patterns
            self.entity_patterns = {
                'object': [
                    r'\b(red|blue|green|yellow|white|black)\s+(\w+)',
                    r'\b(big|small|large|tiny|tall|short)\s+(\w+)',
                    r'\b(\w+)\s+(box|ball|cup|book|chair|table|person|robot)'
                ],
                'location': [
                    r'\b(to|near|at|by)\s+(the\s+)?(\w+)',
                    r'\b(there|here|over\s+there|over\s+here)'
                ],
                'direction': [
                    r'\b(left|right|forward|backward|ahead|behind|up|down)'
                ]
            }

            self.get_logger().info('Language components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize language components: {str(e)}')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data.lower()
        self.get_logger().info(f'Received command: "{self.current_command}"')

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def language_processing_loop(self):
        """Main language processing loop"""
        if self.current_command is None:
            return

        try:
            # Parse command
            parsed_command = self.parse_command(self.current_command)

            # Execute action based on parsed command
            if parsed_command['intent'] in self.intent_actions:
                action_cmd = self.intent_actions[parsed_command['intent']](parsed_command['entities'])
                if action_cmd is not None:
                    self.action_pub.publish(action_cmd)

            # Generate response
            response = self.generate_response(parsed_command)
            if response:
                response_msg = String()
                response_msg.data = response
                self.response_pub.publish(response_msg)

            self.get_logger().info(
                f'Processed command: "{self.current_command}", '
                f'Intent: {parsed_command["intent"]}, '
                f'Entities: {parsed_command["entities"]}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in language processing: {str(e)}')

    def parse_command(self, command: str) -> Dict:
        """Parse command to extract intent and entities"""
        # Normalize command
        command = command.strip().lower()

        # Extract intent
        intent = self.classify_intent(command)

        # Extract entities
        entities = self.extract_entities(command)

        return {
            'intent': intent,
            'entities': entities,
            'original_command': command
        }

    def classify_intent(self, command: str) -> str:
        """Classify intent from command"""
        # Simple keyword-based classification (in real implementation, use ML)
        for intent, keywords in self.intent_classifier.items():
            for keyword in keywords:
                if keyword in command:
                    return intent

        # Default intent if no match found
        return 'move_forward'

    def extract_entities(self, command: str) -> Dict[str, List[str]]:
        """Extract entities from command"""
        entities = {
            'objects': [],
            'locations': [],
            'directions': [],
            'numbers': []
        }

        # Extract objects
        for pattern in self.entity_patterns['object']:
            matches = re.findall(pattern, command)
            for match in matches:
                if isinstance(match, tuple):
                    entities['objects'].extend(list(match))
                else:
                    entities['objects'].append(match)

        # Extract locations
        for pattern in self.entity_patterns['location']:
            matches = re.findall(pattern, command)
            for match in matches:
                if isinstance(match, tuple):
                    entities['locations'].extend([m for m in match if m])
                else:
                    entities['locations'].append(match)

        # Extract directions
        for pattern in self.entity_patterns['direction']:
            matches = re.findall(pattern, command)
            entities['directions'].extend(matches)

        # Extract numbers
        number_pattern = r'\b(\d+)\b'
        number_matches = re.findall(number_pattern, command)
        entities['numbers'] = [int(n) for n in number_matches]

        # Remove empty strings and clean up
        for key in entities:
            entities[key] = [entity for entity in entities[key] if entity.strip()]

        return entities

    def move_forward(self, entities: Dict) -> Twist:
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = 0.0
        return cmd

    def move_backward(self, entities: Dict) -> Twist:
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.3
        cmd.angular.z = 0.0
        return cmd

    def turn_left(self, entities: Dict) -> Twist:
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5
        return cmd

    def turn_right(self, entities: Dict) -> Twist:
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5
        return cmd

    def approach_object(self, entities: Dict) -> Twist:
        """Approach specified object"""
        cmd = Twist()

        # If object is specified, try to find it in detections
        if entities['objects'] and self.current_detections:
            target_object = entities['objects'][0]  # Use first object
            # In a real implementation, you'd search for this object in detections
            cmd.linear.x = 0.2  # Move slowly toward object
        else:
            # Default approach behavior
            cmd.linear.x = 0.2

        cmd.angular.z = 0.0
        return cmd

    def avoid_object(self, entities: Dict) -> Twist:
        """Avoid specified object"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.3  # Turn to avoid
        return cmd

    def stop_robot(self, entities: Dict) -> Twist:
        """Stop robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def follow_object(self, entities: Dict) -> Twist:
        """Follow specified object"""
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        return cmd

    def pick_up_object(self, entities: Dict) -> Twist:
        """Pick up object"""
        cmd = Twist()
        cmd.linear.x = 0.1  # Move slowly to object
        cmd.angular.z = 0.0
        return cmd

    def put_down_object(self, entities: Dict) -> Twist:
        """Put down object"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def generate_response(self, parsed_command: Dict) -> str:
        """Generate response to acknowledge command"""
        intent = parsed_command['intent']
        entities = parsed_command['entities']

        responses = {
            'move_forward': 'Moving forward.',
            'move_backward': 'Moving backward.',
            'turn_left': 'Turning left.',
            'turn_right': 'Turning right.',
            'approach_object': f'Approaching {entities["objects"][0] if entities["objects"] else "object"}.',
            'avoid_object': 'Avoiding object.',
            'stop': 'Stopping.',
            'follow_object': f'Following {entities["objects"][0] if entities["objects"] else "object"}.',
            'pick_up': f'Picking up {entities["objects"][0] if entities["objects"] else "object"}.',
            'put_down': 'Putting down object.'
        }

        return responses.get(intent, 'Processing command.')

def main(args=None):
    rclpy.init(args=args)
    lang_proc = LanguageProcessor()

    try:
        rclpy.spin(lang_proc)
    except KeyboardInterrupt:
        lang_proc.get_logger().info('Language processor shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        lang_proc.action_pub.publish(cmd)

        lang_proc.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Language Understanding with Neural Networks

Implementing more sophisticated language understanding using neural networks:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json

class AdvancedLanguageProcessor(Node):
    def __init__(self):
        super().__init__('advanced_language_processor')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            '/parsed_intent',
            10
        )

        # Initialize advanced language models
        self.tokenizer = None
        self.language_model = None
        self.intent_classifier = None
        self.initialize_advanced_models()

        # State variables
        self.current_command = None
        self.current_detections = None

        # Vocabulary for tokenization
        self.vocab = {}
        self.idx_to_word = {}
        self.max_seq_length = 32

        # Control timer
        self.advanced_timer = self.create_timer(0.1, self.advanced_language_loop)

        self.get_logger().info('Advanced language processor initialized')

    def initialize_advanced_models(self):
        """Initialize advanced language processing models (simplified)"""
        try:
            # Simple LSTM-based language model
            class LanguageModel(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_layers=2):
                    super(LanguageModel, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    return lstm_out

            # Intent classifier
            class IntentClassifier(nn.Module):
                def __init__(self, input_dim=256, num_intents=10):
                    super(IntentClassifier, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.output = nn.Linear(64, num_intents)
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    # Use the last output of the LSTM
                    x = x[:, -1, :]  # [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.output(x)
                    return F.log_softmax(x, dim=1)

            # Initialize models
            self.language_model = LanguageModel()
            self.intent_classifier = IntentClassifier()

            # Create simple vocabulary (in a real system, load from pre-trained model)
            self.vocab = {
                '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3,
                'move': 4, 'forward': 5, 'backward': 6, 'turn': 7,
                'left': 8, 'right': 9, 'go': 10, 'to': 11, 'the': 12,
                'approach': 13, 'avoid': 14, 'stop': 15, 'follow': 16,
                'object': 17, 'person': 18, 'robot': 19, 'table': 20,
                'chair': 21, 'box': 22, 'ball': 23, 'red': 24, 'blue': 25
            }

            # Create reverse mapping
            self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

            # Set models to evaluation mode
            self.language_model.eval()
            self.intent_classifier.eval()

            self.get_logger().info('Advanced language models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced models: {str(e)}')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data.lower()
        self.get_logger().info(f'Received command: "{self.current_command}"')

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def advanced_language_loop(self):
        """Main advanced language processing loop"""
        if (self.language_model is None or
            self.current_command is None):
            return

        try:
            # Tokenize and encode command
            tokenized_command = self.tokenize_command(self.current_command)
            command_tensor = self.encode_command(tokenized_command)

            # Process through language model
            with torch.no_grad():
                features = self.language_model(command_tensor)
                intent_logits = self.intent_classifier(features)

            # Get predicted intent
            predicted_intent = torch.argmax(intent_logits, dim=1).item()
            intent_name = self.get_intent_name(predicted_intent)

            # Execute action based on predicted intent
            action_cmd = self.execute_intent(intent_name)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            # Publish intent
            intent_msg = String()
            intent_msg.data = intent_name
            self.intent_pub.publish(intent_msg)

            self.get_logger().info(
                f'Advanced NLP - Command: "{self.current_command}", '
                f'Predicted Intent: {intent_name} (ID: {predicted_intent})'
            )

        except Exception as e:
            self.get_logger().error(f'Error in advanced language processing: {str(e)}')

    def tokenize_command(self, command: str) -> List[str]:
        """Tokenize command into words"""
        # Simple tokenization
        tokens = command.lower().split()
        # Add start and end tokens
        tokens = ['<START>'] + tokens + ['<END>']
        return tokens

    def encode_command(self, tokens: List[str]) -> torch.Tensor:
        """Encode tokens into tensor"""
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(self.vocab['<UNK>'])

        # Pad or truncate to max sequence length
        if len(indices) < self.max_seq_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_seq_length - len(indices)))
        else:
            indices = indices[:self.max_seq_length]

        # Convert to tensor
        tensor = torch.LongTensor([indices])  # Add batch dimension
        return tensor

    def get_intent_name(self, intent_id: int) -> str:
        """Get intent name from ID"""
        intent_map = {
            0: 'move_forward',
            1: 'move_backward',
            2: 'turn_left',
            3: 'turn_right',
            4: 'approach_object',
            5: 'avoid_object',
            6: 'stop',
            7: 'follow_object',
            8: 'pick_up',
            9: 'put_down'
        }
        return intent_map.get(intent_id, 'unknown')

    def execute_intent(self, intent_name: str) -> Twist:
        """Execute action based on intent"""
        cmd = Twist()

        if intent_name == 'move_forward':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif intent_name == 'move_backward':
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
        elif intent_name == 'turn_left':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif intent_name == 'turn_right':
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif intent_name == 'approach_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent_name == 'avoid_object':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
        elif intent_name == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif intent_name == 'follow_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent_name == 'pick_up':
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
        elif intent_name == 'put_down':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Default: move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    advanced_lang = AdvancedLanguageProcessor()

    try:
        rclpy.spin(advanced_lang)
    except KeyboardInterrupt:
        advanced_lang.get_logger().info('Advanced language processor shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_lang.action_pub.publish(cmd)

        advanced_lang.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Dialogue Management System

Implementing a dialogue management system for multi-turn conversations:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

class DialogueManager(Node):
    def __init__(self):
        super().__init__('dialogue_manager')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.response_pub = self.create_publisher(
            String,
            '/robot_response',
            10
        )

        # Initialize dialogue components
        self.conversation_context = {}
        self.dialogue_state = 'idle'
        self.active_intent = None
        self.pending_requests = []
        self.user_preferences = {}

        # Dialogue patterns and responses
        self.dialogue_patterns = {
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon', r'good evening'
            ],
            'farewell': [
                r'bye', r'goodbye', r'see you', r'talk to you later', r'have a good day'
            ],
            'confirmation': [
                r'yes', r'yep', r'yeah', r'ok', r'okay', r'correct', r'right'
            ],
            'denial': [
                r'no', r'nope', r'wrong', r'incorrect', r'cancel'
            ],
            'request_repeat': [
                r'what', r'pardon', r'excuse me', r'could you repeat', r'say that again'
            ],
            'acknowledgment': [
                r'thanks', r'thank you', r'great', r'awesome', r'perfect'
            ]
        }

        # Intent handlers
        self.intent_handlers = {
            'move_forward': self.handle_move_forward,
            'move_backward': self.handle_move_backward,
            'turn_left': self.handle_turn_left,
            'turn_right': self.handle_turn_right,
            'approach_object': self.handle_approach_object,
            'stop': self.handle_stop,
            'follow_object': self.handle_follow_object
        }

        # State handlers
        self.state_handlers = {
            'idle': self.handle_idle_state,
            'awaiting_confirmation': self.handle_awaiting_confirmation,
            'executing_action': self.handle_executing_action,
            'requesting_clarification': self.handle_requesting_clarification
        }

        # Control timer
        self.dialogue_timer = self.create_timer(0.1, self.dialogue_processing_loop)

        self.get_logger().info('Dialogue manager initialized')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data.lower()
        self.get_logger().info(f'Received command: "{self.current_command}"')

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def dialogue_processing_loop(self):
        """Main dialogue processing loop"""
        if self.current_command is None:
            return

        try:
            # Identify dialogue act
            dialogue_act = self.identify_dialogue_act(self.current_command)

            # Update conversation context
            self.update_conversation_context(dialogue_act)

            # Process command based on current dialogue state
            response = self.process_command_in_context(dialogue_act)

            # Generate robot response
            if response:
                response_msg = String()
                response_msg.data = response
                self.response_pub.publish(response_msg)

            self.get_logger().info(
                f'Dialogue - Command: "{self.current_command}", '
                f'Dialogue Act: {dialogue_act}, State: {self.dialogue_state}'
            )

            # Clear command for next iteration
            self.current_command = None

        except Exception as e:
            self.get_logger().error(f'Error in dialogue processing: {str(e)}')

    def identify_dialogue_act(self, command: str) -> str:
        """Identify the dialogue act of the command"""
        command_lower = command.lower()

        # Check for greetings
        for pattern in self.dialogue_patterns['greeting']:
            if re.search(pattern, command_lower):
                return 'greeting'

        # Check for farewells
        for pattern in self.dialogue_patterns['farewell']:
            if re.search(pattern, command_lower):
                return 'farewell'

        # Check for confirmations
        for pattern in self.dialogue_patterns['confirmation']:
            if re.search(pattern, command_lower):
                return 'confirmation'

        # Check for denials
        for pattern in self.dialogue_patterns['denial']:
            if re.search(pattern, command_lower):
                return 'denial'

        # Check for repeat requests
        for pattern in self.dialogue_patterns['request_repeat']:
            if re.search(pattern, command_lower):
                return 'request_repeat'

        # Check for acknowledgments
        for pattern in self.dialogue_patterns['acknowledgment']:
            if re.search(pattern, command_lower):
                return 'acknowledgment'

        # Check for specific intents
        if any(word in command_lower for word in ['move', 'go', 'forward', 'ahead']):
            return 'move_forward'
        elif any(word in command_lower for word in ['backward', 'back', 'reverse']):
            return 'move_backward'
        elif any(word in command_lower for word in ['turn', 'left']):
            return 'turn_left'
        elif any(word in command_lower for word in ['turn', 'right']):
            return 'turn_right'
        elif any(word in command_lower for word in ['approach', 'go to', 'toward']):
            return 'approach_object'
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return 'stop'
        elif any(word in command_lower for word in ['follow', 'chase', 'track']):
            return 'follow_object'

        # Default: unknown intent
        return 'unknown'

    def update_conversation_context(self, dialogue_act: str):
        """Update conversation context based on dialogue act"""
        # Store last command and act
        self.conversation_context['last_command'] = self.current_command
        self.conversation_context['last_dialogue_act'] = dialogue_act
        self.conversation_context['last_timestamp'] = datetime.now()

        # Update dialogue state based on act
        if dialogue_act in ['greeting', 'farewell', 'acknowledgment']:
            self.dialogue_state = 'idle'
        elif dialogue_act in ['confirmation', 'denial']:
            if self.dialogue_state == 'awaiting_confirmation':
                self.dialogue_state = 'idle'
        elif dialogue_act in ['request_repeat']:
            self.dialogue_state = 'idle'  # Reset to idle to repeat

    def process_command_in_context(self, dialogue_act: str) -> Optional[str]:
        """Process command considering current dialogue context"""
        # Handle the current dialogue state
        if self.dialogue_state in self.state_handlers:
            return self.state_handlers[self.dialogue_state](dialogue_act)
        else:
            # Default handling
            return self.handle_default_state(dialogue_act)

    def handle_idle_state(self, dialogue_act: str) -> Optional[str]:
        """Handle commands when in idle state"""
        if dialogue_act in ['greeting']:
            response = "Hello! How can I assist you today?"
            self.dialogue_state = 'idle'
            return response

        elif dialogue_act in ['farewell']:
            response = "Goodbye! Have a great day!"
            self.dialogue_state = 'idle'
            return response

        elif dialogue_act in ['move_forward', 'move_backward', 'turn_left', 'turn_right',
                             'approach_object', 'stop', 'follow_object']:
            # Execute the action
            action_cmd = self.execute_intent(dialogue_act)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            # Generate response
            responses = {
                'move_forward': "Moving forward.",
                'move_backward': "Moving backward.",
                'turn_left': "Turning left.",
                'turn_right': "Turning right.",
                'approach_object': "Approaching the object.",
                'stop': "Stopping.",
                'follow_object': "Following the object."
            }
            response = responses.get(dialogue_act, "Executing command.")

            self.dialogue_state = 'executing_action'
            return response

        elif dialogue_act in ['acknowledgment']:
            return "Thank you! How else can I help?"

        elif dialogue_act in ['request_repeat']:
            last_command = self.conversation_context.get('last_command', 'previous command')
            return f"You said: '{last_command}'. How can I help?"

        else:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

    def handle_awaiting_confirmation(self, dialogue_act: str) -> Optional[str]:
        """Handle commands when awaiting confirmation"""
        if dialogue_act == 'confirmation':
            # Execute the pending action
            if self.pending_requests:
                pending_request = self.pending_requests.pop(0)
                intent = pending_request['intent']

                action_cmd = self.execute_intent(intent)
                if action_cmd is not None:
                    self.action_pub.publish(action_cmd)

                self.dialogue_state = 'executing_action'
                return f"Okay, {intent.replace('_', ' ')}."

        elif dialogue_act == 'denial':
            # Cancel the pending action
            self.pending_requests.clear()
            self.dialogue_state = 'idle'
            return "Action cancelled. What would you like me to do?"

        else:
            return "Please confirm or deny the previous request."

    def handle_executing_action(self, dialogue_act: str) -> Optional[str]:
        """Handle commands when executing an action"""
        if dialogue_act == 'stop':
            # Stop current action
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.action_pub.publish(stop_cmd)

            self.dialogue_state = 'idle'
            return "Stopped. What would you like to do next?"

        elif dialogue_act in ['move_forward', 'move_backward', 'turn_left', 'turn_right',
                             'approach_object', 'follow_object']:
            # Execute new action, interrupting current one
            action_cmd = self.execute_intent(dialogue_act)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            self.dialogue_state = 'executing_action'
            return f"Changing action to {dialogue_act.replace('_', ' ')}."

        else:
            return "I'm currently executing an action. Would you like me to stop?"

    def handle_requesting_clarification(self, dialogue_act: str) -> Optional[str]:
        """Handle commands when requesting clarification"""
        # In a real implementation, you'd have specific clarification logic
        # For now, just return to idle
        self.dialogue_state = 'idle'
        return "I need more information. Could you please clarify?"

    def handle_default_state(self, dialogue_act: str) -> Optional[str]:
        """Handle commands in default state"""
        return self.handle_idle_state(dialogue_act)

    def execute_intent(self, intent_name: str) -> Twist:
        """Execute action based on intent"""
        cmd = Twist()

        if intent_name == 'move_forward':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif intent_name == 'move_backward':
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
        elif intent_name == 'turn_left':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif intent_name == 'turn_right':
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif intent_name == 'approach_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent_name == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif intent_name == 'follow_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Default: move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

    def handle_move_forward(self, entities: Dict) -> Twist:
        """Handle move forward intent"""
        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = 0.0
        return cmd

    def handle_move_backward(self, entities: Dict) -> Twist:
        """Handle move backward intent"""
        cmd = Twist()
        cmd.linear.x = -0.3
        cmd.angular.z = 0.0
        return cmd

    def handle_turn_left(self, entities: Dict) -> Twist:
        """Handle turn left intent"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5
        return cmd

    def handle_turn_right(self, entities: Dict) -> Twist:
        """Handle turn right intent"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5
        return cmd

    def handle_approach_object(self, entities: Dict) -> Twist:
        """Handle approach object intent"""
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        return cmd

    def handle_stop(self, entities: Dict) -> Twist:
        """Handle stop intent"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def handle_follow_object(self, entities: Dict) -> Twist:
        """Handle follow object intent"""
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        return cmd

def main(args=None):
    rclpy.init(args=args)
    dialogue_manager = DialogueManager()

    try:
        rclpy.spin(dialogue_manager)
    except KeyboardInterrupt:
        dialogue_manager.get_logger().info('Dialogue manager shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        dialogue_manager.action_pub.publish(cmd)

        dialogue_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Context-Aware Language Understanding

Implementing context-aware language understanding for more natural interactions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta

class ContextAwareLanguageProcessor(Node):
    def __init__(self):
        super().__init__('context_aware_language_processor')

        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.response_pub = self.create_publisher(
            String,
            '/robot_response',
            10
        )

        # Initialize context-aware components
        self.conversation_history = []
        self.spatial_context = {}
        self.temporal_context = {}
        self.object_context = {}
        self.location_context = {}
        self.user_preferences = {}

        # Context-sensitive patterns
        self.context_patterns = {
            'relative_direction': [
                r'that way', r'over there', r'this way', r'back there',
                r'over here', r'in front', r'behind', r'beside', r'near'
            ],
            'temporal_reference': [
                r'now', r'again', r'earlier', r'later', r'before', r'after',
                r'just now', r'recently', r'previously'
            ],
            'deictic_reference': [
                r'this', r'that', r'there', r'here', r'it', r'them'
            ],
            'spatial_reference': [
                r'left', r'right', r'forward', r'backward', r'ahead',
                r'above', r'below', r'north', r'south', r'east', r'west'
            ]
        }

        # Reference resolution
        self.referent_resolver = None
        self.initialize_referent_resolver()

        # Control timer
        self.context_timer = self.create_timer(0.1, self.context_aware_processing)

        self.get_logger().info('Context-aware language processor initialized')

    def initialize_referent_resolver(self):
        """Initialize referent resolution components"""
        try:
            # In a real implementation, this would use more sophisticated NLP
            # For now, we'll implement simple heuristics
            self.referent_resolver = {
                'last_mentioned_object': None,
                'last_mentioned_location': None,
                'last_mentioned_direction': None
            }

            self.get_logger().info('Referent resolver initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize referent resolver: {str(e)}')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data.lower()
        self.get_logger().info(f'Received command: "{self.current_command}"')

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

        # Update object context with current detections
        self.update_object_context(msg)

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.current_scan = msg

    def context_aware_processing(self):
        """Main context-aware language processing loop"""
        if self.current_command is None:
            return

        try:
            # Parse command with context
            parsed_command = self.parse_command_with_context(self.current_command)

            # Resolve references
            resolved_command = self.resolve_references(parsed_command)

            # Execute action
            action_cmd = self.execute_resolved_command(resolved_command)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            # Generate response
            response = self.generate_contextual_response(resolved_command)
            if response:
                response_msg = String()
                response_msg.data = response
                self.response_pub.publish(response_msg)

            # Update context
            self.update_context(parsed_command, resolved_command)

            self.get_logger().info(
                f'Context-aware processing - Command: "{self.current_command}", '
                f'Resolved: "{resolved_command}", Action executed'
            )

            # Clear command for next iteration
            self.current_command = None

        except Exception as e:
            self.get_logger().error(f'Error in context-aware processing: {str(e)}')

    def parse_command_with_context(self, command: str) -> Dict:
        """Parse command considering context"""
        # Identify patterns in the command
        relative_directions = []
        temporal_refs = []
        deictic_refs = []
        spatial_refs = []

        for pattern_list, ref_list in [
            (self.context_patterns['relative_direction'], relative_directions),
            (self.context_patterns['temporal_reference'], temporal_refs),
            (self.context_patterns['deictic_reference'], deictic_refs),
            (self.context_patterns['spatial_reference'], spatial_refs)
        ]:
            for pattern in pattern_list:
                if re.search(pattern, command):
                    ref_list.append(pattern)

        # Identify intent
        intent = self.identify_intent_from_context(command)

        return {
            'original_command': command,
            'relative_directions': relative_directions,
            'temporal_references': temporal_refs,
            'deictic_references': deictic_refs,
            'spatial_references': spatial_refs,
            'intent': intent,
            'resolved_entities': {}
        }

    def identify_intent_from_context(self, command: str) -> str:
        """Identify intent considering context"""
        # Start with basic intent identification
        basic_intent = self.basic_intent_identification(command)

        # Enhance with context
        if 'there' in command or 'that' in command:
            # Could be referring to something previously mentioned
            if self.referent_resolver['last_mentioned_object']:
                return 'approach_object'
            elif self.referent_resolver['last_mentioned_location']:
                return 'go_to_location'

        # Check for relative spatial references
        if any(ref in command for ref in self.context_patterns['relative_direction']):
            return 'navigate_relative'

        return basic_intent

    def basic_intent_identification(self, command: str) -> str:
        """Basic intent identification"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['move', 'go', 'forward', 'ahead']):
            return 'move_forward'
        elif any(word in command_lower for word in ['backward', 'back', 'reverse']):
            return 'move_backward'
        elif any(word in command_lower for word in ['turn', 'left']):
            return 'turn_left'
        elif any(word in command_lower for word in ['turn', 'right']):
            return 'turn_right'
        elif any(word in command_lower for word in ['approach', 'go to', 'toward']):
            return 'approach_object'
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return 'stop'
        elif any(word in command_lower for word in ['follow', 'chase', 'track']):
            return 'follow_object'
        else:
            return 'unknown'

    def resolve_references(self, parsed_command: Dict) -> Dict:
        """Resolve deictic and anaphoric references"""
        resolved = parsed_command.copy()

        # Resolve 'it', 'that', 'this', 'there' references
        for ref in parsed_command['deictic_references']:
            if ref in ['it', 'that']:
                # Resolve to last mentioned object
                if self.referent_resolver['last_mentioned_object']:
                    resolved['resolved_entities']['object'] = self.referent_resolver['last_mentioned_object']
            elif ref in ['this', 'here']:
                # Resolve to current location or nearby object
                if self.current_detections:
                    # Use closest object as reference
                    closest_obj = self.get_closest_object()
                    if closest_obj:
                        resolved['resolved_entities']['object'] = closest_obj

        # Resolve spatial references with context
        for spatial_ref in parsed_command['spatial_references']:
            if spatial_ref in ['left', 'right', 'forward', 'backward']:
                resolved['resolved_entities']['direction'] = spatial_ref

        # Resolve temporal references
        for temp_ref in parsed_command['temporal_references']:
            if temp_ref in ['now', 'again']:
                # Repeat last action
                resolved['resolved_entities']['repeat'] = True

        return resolved

    def get_closest_object(self):
        """Get the closest detected object"""
        if not self.current_detections:
            return None

        # In a real implementation, you'd have access to distance information
        # For this example, we'll just return the first detection
        if self.current_detections.detections:
            return self.current_detections.detections[0]

        return None

    def execute_resolved_command(self, resolved_command: Dict) -> Optional[Twist]:
        """Execute the resolved command"""
        intent = resolved_command['intent']

        if intent == 'move_forward':
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            return cmd
        elif intent == 'move_backward':
            cmd = Twist()
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
            return cmd
        elif intent == 'turn_left':
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
            return cmd
        elif intent == 'turn_right':
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
            return cmd
        elif intent == 'approach_object':
            cmd = Twist()
            # If object is specified, approach it
            if 'object' in resolved_command['resolved_entities']:
                cmd.linear.x = 0.2
            else:
                cmd.linear.x = 0.2  # Default approach
            cmd.angular.z = 0.0
            return cmd
        elif intent == 'stop':
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        elif intent == 'navigate_relative':
            cmd = Twist()
            # Handle relative navigation based on context
            direction = resolved_command['resolved_entities'].get('direction', 'forward')
            if direction == 'left':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
            elif direction == 'right':
                cmd.linear.x = 0.0
                cmd.angular.z = -0.5
            elif direction == 'forward':
                cmd.linear.x = 0.3
                cmd.angular.z = 0.0
            elif direction == 'backward':
                cmd.linear.x = -0.3
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.3
                cmd.angular.z = 0.0
            return cmd
        else:
            # Default action
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            return cmd

    def generate_contextual_response(self, resolved_command: Dict) -> str:
        """Generate response considering context"""
        intent = resolved_command['intent']

        responses = {
            'move_forward': "Moving forward.",
            'move_backward': "Moving backward.",
            'turn_left': "Turning left.",
            'turn_right': "Turning right.",
            'approach_object': f"Approaching {'object' if 'object' in resolved_command['resolved_entities'] else 'the object'}.",
            'stop': "Stopping.",
            'navigate_relative': f"Navigating {resolved_command['resolved_entities'].get('direction', 'forward')}."
        }

        return responses.get(intent, "Executing command.")

    def update_context(self, parsed_command: Dict, resolved_command: Dict):
        """Update various contexts based on command and resolution"""
        # Update conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'command': parsed_command,
            'resolution': resolved_command
        })

        # Trim history to keep only recent entries (last 10)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # Update referent resolver
        if 'object' in resolved_command['resolved_entities']:
            self.referent_resolver['last_mentioned_object'] = resolved_command['resolved_entities']['object']

        # Update spatial context
        if 'direction' in resolved_command['resolved_entities']:
            self.referent_resolver['last_mentioned_direction'] = resolved_command['resolved_entities']['direction']

    def update_object_context(self, detections: Detection2DArray):
        """Update object context with current detections"""
        for detection in detections.detections:
            # Store object information
            obj_id = f"obj_{len(self.object_context)}"
            self.object_context[obj_id] = {
                'bbox': detection.bbox,
                'results': detection.results,
                'timestamp': datetime.now()
            }

        # Clean up old object context (keep only recent objects)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.object_context = {
            k: v for k, v in self.object_context.items()
            if v['timestamp'] > cutoff_time
        }

def main(args=None):
    rclpy.init(args=args)
    context_lang = ContextAwareLanguageProcessor()

    try:
        rclpy.spin(context_lang)
    except KeyboardInterrupt:
        context_lang.get_logger().info('Context-aware language processor shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        context_lang.action_pub.publish(cmd)

        context_lang.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Integration with Vision

Integrating voice commands with vision processing for enhanced interaction:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class VoiceVisionIntegration(Node):
    def __init__(self):
        super().__init__('voice_vision_integration')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.response_pub = self.create_publisher(
            String,
            '/robot_response',
            10
        )

        # Initialize components
        self.voice_processor = None
        self.vision_processor = None
        self.vision_voice_fusion = None
        self.initialize_integration_components()

        # State variables
        self.current_command = None
        self.current_image = None
        self.current_detections = None
        self.command_vision_context = {}

        # Control timer
        self.integration_timer = self.create_timer(0.1, self.voice_vision_integration_loop)

        self.get_logger().info('Voice-vision integration initialized')

    def initialize_integration_components(self):
        """Initialize integration components"""
        try:
            # In a real implementation, this would include sophisticated fusion models
            # For this example, we'll implement simple fusion logic

            # Voice command interpretation with vision context
            class VoiceVisionFusion:
                def __init__(self):
                    self.vision_context = {}
                    self.voice_context = {}

                def fuse_interpretation(self, command: str, vision_data: dict) -> Dict:
                    """Fuse voice command with vision data for interpretation"""
                    result = {
                        'command': command,
                        'vision_data': vision_data,
                        'fused_intent': '',
                        'target_object': None,
                        'target_location': None
                    }

                    # Simple fusion logic
                    if 'red' in command.lower() and vision_data.get('objects'):
                        # Find red object in vision data
                        for obj in vision_data['objects']:
                            if 'red' in obj.get('attributes', []):
                                result['target_object'] = obj
                                result['fused_intent'] = 'approach_object'
                                break

                    elif 'person' in command.lower() and vision_data.get('objects'):
                        # Find person in vision data
                        for obj in vision_data['objects']:
                            if obj.get('class') == 'person':
                                result['target_object'] = obj
                                result['fused_intent'] = 'follow_object'
                                break

                    elif 'box' in command.lower() and vision_data.get('objects'):
                        # Find box in vision data
                        for obj in vision_data['objects']:
                            if 'box' in obj.get('class', '').lower():
                                result['target_object'] = obj
                                result['fused_intent'] = 'approach_object'
                                break

                    # Default to basic interpretation if no vision context
                    if not result['fused_intent']:
                        if any(word in command.lower() for word in ['move', 'go', 'forward']):
                            result['fused_intent'] = 'move_forward'
                        elif any(word in command.lower() for word in ['stop', 'halt']):
                            result['fused_intent'] = 'stop'
                        elif any(word in command.lower() for word in ['turn', 'left']):
                            result['fused_intent'] = 'turn_left'
                        elif any(word in command.lower() for word in ['turn', 'right']):
                            result['fused_intent'] = 'turn_right'

                    return result

            self.vision_voice_fusion = VoiceVisionFusion()
            self.get_logger().info('Voice-vision fusion component initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize integration components: {str(e)}')

    def command_callback(self, msg):
        """Process voice command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received voice command: "{self.current_command}"')

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def voice_vision_integration_loop(self):
        """Main voice-vision integration loop"""
        if (self.current_command is None or
            self.current_image is None or
            self.vision_voice_fusion is None):
            return

        try:
            # Prepare vision data from detections
            vision_data = self.extract_vision_data()

            # Fuse voice command with vision data
            fused_result = self.vision_voice_fusion.fuse_interpretation(
                self.current_command, vision_data
            )

            # Execute action based on fused interpretation
            action_cmd = self.execute_fused_action(fused_result)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            # Generate response
            response = self.generate_fusion_response(fused_result)
            if response:
                response_msg = String()
                response_msg.data = response
                self.response_pub.publish(response_msg)

            self.get_logger().info(
                f'Voice-Vision Fusion - Command: "{self.current_command}", '
                f'Intent: {fused_result["fused_intent"]}, '
                f'Target: {fused_result["target_object"]["class"] if fused_result["target_object"] else "None"}'
            )

            # Clear command for next iteration
            self.current_command = None

        except Exception as e:
            self.get_logger().error(f'Error in voice-vision integration: {str(e)}')

    def extract_vision_data(self) -> Dict:
        """Extract structured vision data from detections"""
        vision_data = {
            'objects': [],
            'scene_description': '',
            'spatial_relationships': []
        }

        if self.current_detections:
            for detection in self.current_detections.detections:
                obj_data = {
                    'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                    'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                    'bbox': {
                        'center_x': detection.bbox.center.x,
                        'center_y': detection.bbox.center.y,
                        'width': detection.bbox.size_x,
                        'height': detection.bbox.size_y
                    },
                    'attributes': []  # Could include color, size, etc.
                }

                # Extract color information if available
                if 'red' in obj_data['class'] or 'blue' in obj_data['class']:
                    obj_data['attributes'].append(obj_data['class'].split()[0])

                vision_data['objects'].append(obj_data)

        return vision_data

    def execute_fused_action(self, fused_result: Dict) -> Optional[Twist]:
        """Execute action based on fused interpretation"""
        intent = fused_result['fused_intent']

        cmd = Twist()

        if intent == 'move_forward':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif intent == 'move_backward':
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
        elif intent == 'turn_left':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif intent == 'turn_right':
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif intent == 'approach_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent == 'follow_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Default action
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

    def generate_fusion_response(self, fused_result: Dict) -> str:
        """Generate response based on fused interpretation"""
        intent = fused_result['fused_intent']
        target_obj = fused_result['target_object']

        if target_obj:
            object_desc = f"{target_obj['class']} at ({target_obj['bbox']['center_x']:.0f}, {target_obj['bbox']['center_y']:.0f})"
            responses = {
                'move_forward': f"Moving forward.",
                'move_backward': f"Moving backward.",
                'turn_left': f"Turning left.",
                'turn_right': f"Turning right.",
                'approach_object': f"Approaching the {object_desc}.",
                'follow_object': f"Following the {object_desc}.",
                'stop': f"Stopping."
            }
        else:
            responses = {
                'move_forward': "Moving forward.",
                'move_backward': "Moving backward.",
                'turn_left': "Turning left.",
                'turn_right': "Turning right.",
                'approach_object': "Approaching object.",
                'follow_object': "Following object.",
                'stop': "Stopping."
            }

        return responses.get(intent, "Processing command.")

def main(args=None):
    rclpy.init(args=args)
    voice_vision = VoiceVisionIntegration()

    try:
        rclpy.spin(voice_vision)
    except KeyboardInterrupt:
        voice_vision.get_logger().info('Voice-vision integration shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        voice_vision.action_pub.publish(cmd)

        voice_vision.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Language Understanding

1. **Context Awareness**: Maintain conversation and spatial context
2. **Robust Parsing**: Handle ambiguous or incomplete commands gracefully
3. **Multi-turn Support**: Enable natural, flowing conversations
4. **Error Recovery**: Provide helpful feedback when commands are misunderstood
5. **User Adaptation**: Learn user preferences and speaking patterns
6. **Privacy Considerations**: Handle sensitive information appropriately
7. **Latency Optimization**: Process commands quickly for natural interaction
8. **Fallback Strategies**: Have backup plans when primary understanding fails

### Physical Grounding and Simulation-to-Real Mapping

When implementing language understanding for robotics:

- **Speech Recognition**: Account for real-world acoustic conditions
- **Ambient Noise**: Consider background noise in real environments
- **Distance**: Account for microphone distance and speech clarity
- **Processing Delays**: Consider computational delays in real-time systems
- **Ambiguity Handling**: Develop strategies for resolving ambiguous references
- **Cultural Context**: Consider cultural and linguistic differences
- **Safety Systems**: Implement safety checks around language-driven actions

### Troubleshooting Language Processing Issues

Common language processing problems and solutions:

- **Recognition Errors**: Improve acoustic models and noise reduction
- **Context Loss**: Implement better context tracking and recovery
- **Ambiguity**: Develop disambiguation strategies and user clarification
- **Performance Issues**: Optimize models and consider edge processing
- **Integration Problems**: Ensure proper data flow between components
- **Multilingual Support**: Implement language detection and translation

### Summary

This chapter covered natural language understanding and processing for robotics applications, focusing on how to implement systems that can interpret human commands, engage in conversations, and follow complex instructions expressed in natural language. You learned about basic and advanced language processing techniques, dialogue management, context-aware understanding, and voice-vision integration. Language understanding enables robots to interact naturally with humans, making them more accessible and useful in real-world applications. In the next chapter, we'll explore action planning and execution for robotics applications.