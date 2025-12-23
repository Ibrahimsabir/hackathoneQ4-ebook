# Module 4: Vision–Language–Action (VLA)

## Chapter 4.4: Voice-to-Action Pipelines

This chapter focuses on voice-to-action pipelines that convert spoken language commands into robotic actions. These pipelines integrate speech recognition, natural language understanding, and action execution to enable natural human-robot interaction through voice commands.

### Understanding Voice-to-Action Pipelines

Voice-to-action pipelines are essential for creating intuitive human-robot interfaces. They transform spoken commands into actionable robot behaviors through several stages:

- **Audio Processing**: Capturing and preprocessing audio signals
- **Speech Recognition**: Converting speech to text
- **Natural Language Understanding**: Interpreting the meaning of commands
- **Intent Classification**: Determining the action to be performed
- **Entity Extraction**: Identifying relevant objects, locations, and parameters
- **Action Mapping**: Converting interpreted commands to robot actions
- **Execution**: Performing the specified actions
- **Feedback**: Providing confirmation or clarification

### Voice-to-Action Architecture

The voice-to-action pipeline typically follows this architecture:

```
+-------------------+
|   Audio Input     |
|   (Microphone)    |
+-------------------+
|   Audio Preproc   |
|   (Noise Filter)  |
+-------------------+
|   Speech Recogn.  |
|   (ASR)           |
+-------------------+
|   Text Process.   |
|   (NLU)           |
+-------------------+
|   Intent Class.   |
|   (Classifier)    |
+-------------------+
|   Action Map.     |
|   (ROS Actions)   |
+-------------------+
|   Robot Action    |
|   (Execution)     |
+-------------------+
```

### Basic Voice-to-Action Pipeline

Implementing a foundational voice-to-action pipeline:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int8
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from audio_common_msgs.msg import AudioData
import numpy as np
import speech_recognition as sr
import pyaudio
import threading
import queue
import time
from typing import Dict, List, Optional

class VoiceToActionPipeline(Node):
    def __init__(self):
        super().__init__('voice_to_action_pipeline')

        # Publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/voice_status',
            10
        )

        # Initialize speech recognition components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()

        # Initialize with sample rate and chunk size
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Initialize pipeline components
        self.nlu_processor = None
        self.intent_classifier = None
        self.action_mapper = None
        self.initialize_pipeline_components()

        # State variables
        self.current_audio = None
        self.current_command = None
        self.current_intent = None
        self.pipeline_state = 'listening'  # listening, processing, executing
        self.is_listening = True

        # Pipeline parameters
        self.energy_threshold = 300  # For silence detection
        self.pause_threshold = 0.8   # Seconds of silence to consider end of phrase
        self.timeout = 5.0           # Timeout for recognition

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Control timer
        self.pipeline_timer = self.create_timer(0.1, self.pipeline_processing_loop)

        self.get_logger().info('Voice-to-action pipeline initialized')

    def initialize_pipeline_components(self):
        """Initialize voice-to-action pipeline components"""
        try:
            # Intent classifier (simplified rule-based)
            self.intent_classifier = {
                'move_forward': [
                    'move forward', 'go forward', 'move ahead', 'go ahead',
                    'forward', 'straight', 'ahead', 'go straight', 'move up'
                ],
                'move_backward': [
                    'move backward', 'go backward', 'move back', 'go back',
                    'backward', 'back', 'reverse', 'move down'
                ],
                'turn_left': [
                    'turn left', 'go left', 'turn left', 'left', 'spin left'
                ],
                'turn_right': [
                    'turn right', 'go right', 'turn right', 'right', 'spin right'
                ],
                'approach_object': [
                    'go to', 'move to', 'approach', 'come to', 'go near',
                    'approach the', 'move toward', 'go toward', 'go to the'
                ],
                'avoid_object': [
                    'avoid', 'stay away', 'move away', 'go around', 'steer clear'
                ],
                'stop': [
                    'stop', 'halt', 'pause', 'freeze', 'cease', 'quit', 'hold'
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

            # Action mapper
            self.action_mapper = {
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

            self.get_logger().info('Pipeline components initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize pipeline components: {str(e)}')

    def audio_callback(self, msg):
        """Process audio data"""
        self.current_audio = msg

    def audio_processing_loop(self):
        """Background audio processing loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=5
                    )

                # Put audio in queue for processing
                self.audio_queue.put(audio)

                # Update status
                status_msg = String()
                status_msg.data = 'Audio captured'
                self.status_pub.publish(status_msg)

            except sr.WaitTimeoutError:
                # Timeout occurred, continue listening
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {str(e)}')
                time.sleep(0.1)

    def pipeline_processing_loop(self):
        """Main pipeline processing loop"""
        try:
            # Check for new audio in queue
            while not self.audio_queue.empty():
                audio = self.audio_queue.get()

                # Convert audio to text
                text = self.speech_to_text(audio)

                if text:
                    # Process the command
                    self.process_voice_command(text)

        except Exception as e:
            self.get_logger().error(f'Error in pipeline processing: {str(e)}')

    def speech_to_text(self, audio):
        """Convert speech audio to text"""
        try:
            # Use Google Speech Recognition (requires internet)
            # For offline recognition, you could use pocketsphinx
            text = self.recognizer.recognize_google(audio)
            self.get_logger().info(f'Recognized: "{text}"')
            return text.lower()

        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio')
            return None
        except sr.RequestError as e:
            self.get_logger().error(f'Speech recognition error: {str(e)}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error in speech recognition: {str(e)}')
            return None

    def process_voice_command(self, command_text: str):
        """Process recognized voice command"""
        self.current_command = command_text

        # Publish command
        cmd_msg = String()
        cmd_msg.data = command_text
        self.command_pub.publish(cmd_msg)

        # Classify intent
        intent = self.classify_intent(command_text)

        # Extract entities
        entities = self.extract_entities(command_text)

        # Execute action based on intent
        if intent in self.action_mapper:
            action_cmd = self.action_mapper[intent](entities)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

        # Update status
        status_msg = String()
        status_msg.data = f'Command: "{command_text}", Intent: {intent}'
        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f'Voice Command: "{command_text}", Intent: {intent}, Entities: {entities}'
        )

    def classify_intent(self, command: str) -> str:
        """Classify intent from command text"""
        # Simple keyword-based classification
        for intent, keywords in self.intent_classifier.items():
            for keyword in keywords:
                if keyword in command:
                    return intent

        # Default intent if no match found
        return 'move_forward'

    def extract_entities(self, command: str) -> Dict:
        """Extract entities from command"""
        entities = {
            'objects': [],
            'locations': [],
            'directions': [],
            'numbers': []
        }

        # Extract objects (simple pattern matching)
        object_patterns = [
            r'\b(red|blue|green|yellow|white|black)\s+(\w+)',
            r'\b(big|small|large|tiny|tall|short)\s+(\w+)',
            r'\b(\w+)\s+(box|ball|cup|book|chair|table|person|robot|object)'
        ]

        for pattern in object_patterns:
            import re
            matches = re.findall(pattern, command)
            for match in matches:
                if isinstance(match, tuple):
                    entities['objects'].extend([item for item in match if item.strip()])
                else:
                    entities['objects'].append(match)

        # Extract directions
        directions = ['left', 'right', 'forward', 'backward', 'ahead', 'behind', 'up', 'down']
        for direction in directions:
            if direction in command:
                entities['directions'].append(direction)

        # Extract numbers
        import re
        number_matches = re.findall(r'\b(\d+)\b', command)
        entities['numbers'] = [int(n) for n in number_matches]

        # Clean up
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
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        return cmd

    def avoid_object(self, entities: Dict) -> Twist:
        """Avoid object"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.3
        return cmd

    def stop_robot(self, entities: Dict) -> Twist:
        """Stop robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def follow_object(self, entities: Dict) -> Twist:
        """Follow object"""
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        return cmd

    def pick_up_object(self, entities: Dict) -> Twist:
        """Pick up object"""
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.0
        return cmd

    def put_down_object(self, entities: Dict) -> Twist:
        """Put down object"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

def main(args=None):
    rclpy.init(args=args)
    voice_pipeline = VoiceToActionPipeline()

    try:
        rclpy.spin(voice_pipeline)
    except KeyboardInterrupt:
        voice_pipeline.get_logger().info('Voice-to-action pipeline shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        voice_pipeline.action_pub.publish(cmd)

        # Stop audio processing
        voice_pipeline.is_listening = False

        voice_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Voice-to-Action with Deep Learning

Implementing more sophisticated voice-to-action using deep learning models:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Tuple
import librosa
import threading
import queue

class AdvancedVoiceToAction(Node):
    def __init__(self):
        super().__init__('advanced_voice_to_action')

        # Publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.intent_pub = self.create_publisher(
            String,
            '/predicted_intent',
            10
        )

        # Initialize advanced models
        self.speech_model = None
        self.language_model = None
        self.intent_classifier = None
        self.initialize_advanced_models()

        # Audio processing components
        self.sample_rate = 16000
        self.window_size = 1024
        self.hop_length = 512
        self.n_mfcc = 13
        self.audio_buffer = np.array([])

        # State variables
        self.current_audio = None
        self.current_command = None
        self.current_intent = None

        # Initialize audio processing
        self.audio_queue = queue.Queue()
        self.is_processing = True

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Control timer
        self.advanced_timer = self.create_timer(0.1, self.advanced_processing_loop)

        self.get_logger().info('Advanced voice-to-action pipeline initialized')

    def initialize_advanced_models(self):
        """Initialize advanced deep learning models (simplified)"""
        try:
            # Speech feature extractor (MFCC-based)
            class SpeechFeatureExtractor(nn.Module):
                def __init__(self, sample_rate=16000, n_mfcc=13):
                    super(SpeechFeatureExtractor, self).__init__()
                    self.sample_rate = sample_rate
                    self.n_mfcc = n_mfcc

                def forward(self, audio):
                    # In a real implementation, this would extract MFCC features
                    # For now, we'll simulate feature extraction
                    # Convert audio to MFCC features
                    if len(audio.shape) > 1:
                        audio = audio.squeeze(0)  # Remove batch dimension if present

                    # Compute MFCC features
                    mfccs = librosa.feature.mfcc(y=audio.numpy(), sr=self.sample_rate, n_mfcc=self.n_mfcc)

                    # Pad or truncate to fixed length
                    max_length = 100  # Fixed length for simplicity
                    if mfccs.shape[1] < max_length:
                        # Pad with zeros
                        padded = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
                    else:
                        # Truncate
                        padded = mfccs[:, :max_length]

                    return torch.FloatTensor(padded)

            # Language understanding model
            class LanguageUnderstandingModel(nn.Module):
                def __init__(self, input_dim=13, hidden_dim=128, num_layers=2):
                    super(LanguageUnderstandingModel, self).__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    # x shape: [batch, features, time_steps]
                    x = x.transpose(1, 2)  # [batch, time_steps, features]
                    lstm_out, _ = self.lstm(x)
                    return lstm_out

            # Intent classifier
            class IntentClassifier(nn.Module):
                def __init__(self, input_dim=128, num_intents=10):
                    super(IntentClassifier, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.output = nn.Linear(32, num_intents)
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    # Use the last time step's output
                    x = x[:, -1, :]  # [batch, hidden_dim]
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.output(x)
                    return F.log_softmax(x, dim=1)

            # Initialize models
            self.speech_feature_extractor = SpeechFeatureExtractor()
            self.language_model = LanguageUnderstandingModel()
            self.intent_classifier = IntentClassifier()

            # Set models to evaluation mode
            self.speech_feature_extractor.eval()
            self.language_model.eval()
            self.intent_classifier.eval()

            self.get_logger().info('Advanced models initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize advanced models: {str(e)}')

    def audio_callback(self, msg):
        """Process audio data"""
        # Convert AudioData to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
        audio_tensor = torch.FloatTensor(audio_data)

        # Put audio in processing queue
        self.audio_queue.put(audio_tensor)

    def audio_processing_worker(self):
        """Background audio processing worker"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    audio_tensor = self.audio_queue.get(timeout=1.0)

                    # Process audio through models
                    with torch.no_grad():
                        # Extract features
                        features = self.speech_feature_extractor(audio_tensor.unsqueeze(0))

                        # Process through language model
                        lang_features = self.language_model(features)

                        # Classify intent
                        intent_logits = self.intent_classifier(lang_features)
                        predicted_intent_idx = torch.argmax(intent_logits, dim=1).item()

                        # Convert to intent name
                        intent_name = self.index_to_intent(predicted_intent_idx)

                        # Store result
                        self.current_intent = intent_name

                        # Publish intent
                        intent_msg = String()
                        intent_msg.data = intent_name
                        self.intent_pub.publish(intent_msg)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing worker: {str(e)}')

    def advanced_processing_loop(self):
        """Main advanced processing loop"""
        if self.current_intent is None:
            return

        try:
            # Execute action based on predicted intent
            action_cmd = self.execute_predicted_intent(self.current_intent)
            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

            self.get_logger().info(
                f'Advanced VTA - Intent: {self.current_intent}, '
                f'Action executed'
            )

            # Clear intent for next iteration
            self.current_intent = None

        except Exception as e:
            self.get_logger().error(f'Error in advanced processing: {str(e)}')

    def index_to_intent(self, idx: int) -> str:
        """Convert intent index to name"""
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
        return intent_map.get(idx, 'move_forward')

    def execute_predicted_intent(self, intent_name: str) -> Twist:
        """Execute action based on predicted intent"""
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
    advanced_vta = AdvancedVoiceToAction()

    try:
        rclpy.spin(advanced_vta)
    except KeyboardInterrupt:
        advanced_vta.get_logger().info('Advanced voice-to-action shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        advanced_vta.action_pub.publish(cmd)

        # Stop processing
        advanced_vta.is_processing = False

        advanced_vta.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command with Vision Integration

Implementing voice commands that utilize visual context:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData, Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import speech_recognition as sr
import threading
import queue
from typing import Dict, List, Optional
import re

class VoiceVisionAction(Node):
    def __init__(self):
        super().__init__('voice_vision_action')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
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

        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
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

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize with sample rate and chunk size
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Initialize components
        self.voice_processor = None
        self.vision_processor = None
        self.fusion_processor = None
        self.initialize_vision_voice_components()

        # State variables
        self.current_audio = None
        self.current_image = None
        self.current_detections = None
        self.current_command = None
        self.current_intent = None
        self.vision_context = {}

        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = True

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Control timer
        self.vva_timer = self.create_timer(0.1, self.voice_vision_processing_loop)

        self.get_logger().info('Voice-vision action pipeline initialized')

    def initialize_vision_voice_components(self):
        """Initialize voice and vision processing components"""
        try:
            # Vision-voice fusion processor
            class VisionVoiceFusion:
                def __init__(self):
                    self.object_context = {}
                    self.spatial_context = {}

                def fuse_interpretation(self, command: str, vision_data: Dict) -> Dict:
                    """Fuse voice command with vision data for interpretation"""
                    result = {
                        'command': command,
                        'vision_data': vision_data,
                        'fused_intent': '',
                        'target_object': None,
                        'target_location': None,
                        'resolution_confidence': 0.0
                    }

                    # Enhanced interpretation based on visual context
                    if vision_data.get('objects'):
                        # Look for specific object references in command
                        command_lower = command.lower()

                        # Find objects mentioned in command
                        for obj in vision_data['objects']:
                            obj_class = obj.get('class', '').lower()
                            obj_attributes = obj.get('attributes', [])

                            # Check if object is mentioned in command
                            if obj_class in command_lower or any(attr in command_lower for attr in obj_attributes):
                                result['target_object'] = obj
                                result['resolution_confidence'] = obj.get('confidence', 0.7)

                                # Determine intent based on command
                                if any(word in command_lower for word in ['approach', 'go to', 'move to']):
                                    result['fused_intent'] = 'approach_object'
                                elif any(word in command_lower for word in ['follow', 'chase', 'track']):
                                    result['fused_intent'] = 'follow_object'
                                elif any(word in command_lower for word in ['avoid', 'stay away']):
                                    result['fused_intent'] = 'avoid_object'
                                else:
                                    result['fused_intent'] = 'approach_object'

                                return result

                    # Default interpretation if no visual context matches
                    result['fused_intent'] = self.basic_intent_classification(command)
                    result['resolution_confidence'] = 0.5  # Lower confidence without visual context

                    return result

                def basic_intent_classification(self, command: str) -> str:
                    """Basic intent classification without visual context"""
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
                    elif any(word in command_lower for word in ['avoid', 'stay away']):
                        return 'avoid_object'
                    elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
                        return 'stop'
                    elif any(word in command_lower for word in ['follow', 'chase', 'track']):
                        return 'follow_object'
                    elif any(word in command_lower for word in ['pick up', 'grab', 'take']):
                        return 'pick_up'
                    elif any(word in command_lower for word in ['put down', 'drop', 'place']):
                        return 'put_down'
                    else:
                        return 'move_forward'

            self.fusion_processor = VisionVoiceFusion()
            self.get_logger().info('Vision-voice fusion processor initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize vision-voice components: {str(e)}')

    def audio_callback(self, msg):
        """Process audio data"""
        self.current_audio = msg

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def audio_processing_loop(self):
        """Background audio processing loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=5)

                # Put audio in queue for processing
                self.audio_queue.put(audio)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {str(e)}')
                continue

    def voice_vision_processing_loop(self):
        """Main voice-vision processing loop"""
        try:
            # Process new audio if available
            while not self.audio_queue.empty():
                audio = self.audio_queue.get()

                # Convert speech to text
                command_text = self.speech_to_text(audio)

                if command_text:
                    # Process command with vision context
                    self.process_vision_voice_command(command_text)

        except Exception as e:
            self.get_logger().error(f'Error in voice-vision processing: {str(e)}')

    def speech_to_text(self, audio):
        """Convert speech to text"""
        try:
            text = self.recognizer.recognize_google(audio)
            self.get_logger().info(f'Recognized: "{text}"')
            return text.lower()
        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio')
            return None
        except sr.RequestError as e:
            self.get_logger().error(f'Speech recognition error: {str(e)}')
            return None

    def process_vision_voice_command(self, command_text: str):
        """Process voice command with visual context"""
        # Prepare vision data
        vision_data = self.prepare_vision_data()

        # Fuse voice and vision interpretation
        fused_result = self.fusion_processor.fuse_interpretation(command_text, vision_data)

        # Store current command and intent
        self.current_command = command_text
        self.current_intent = fused_result['fused_intent']

        # Execute action based on fused interpretation
        action_cmd = self.execute_fused_action(fused_result)
        if action_cmd is not None:
            self.action_pub.publish(action_cmd)

        # Generate response
        response = self.generate_vision_voice_response(fused_result)
        if response:
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

        # Publish command
        cmd_msg = String()
        cmd_msg.data = command_text
        self.command_pub.publish(cmd_msg)

        self.get_logger().info(
            f'Vision-Voice Command: "{command_text}", '
            f'Intent: {fused_result["fused_intent"]}, '
            f'Confidence: {fused_result["resolution_confidence"]:.2f}, '
            f'Target: {fused_result["target_object"]["class"] if fused_result["target_object"] else "None"}'
        )

    def prepare_vision_data(self) -> Dict:
        """Prepare vision data for fusion processing"""
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

                # Extract additional attributes if available
                if 'red' in obj_data['class'] or 'blue' in obj_data['class']:
                    obj_data['attributes'].append(obj_data['class'].split()[0])

                vision_data['objects'].append(obj_data)

        return vision_data

    def execute_fused_action(self, fused_result: Dict) -> Twist:
        """Execute action based on fused interpretation"""
        intent = fused_result['fused_intent']
        target_obj = fused_result['target_object']

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
        elif intent == 'avoid_object':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
        elif intent == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif intent == 'follow_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif intent == 'pick_up':
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
        elif intent == 'put_down':
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

    def generate_vision_voice_response(self, fused_result: Dict) -> str:
        """Generate response based on fused interpretation"""
        intent = fused_result['fused_intent']
        target_obj = fused_result['target_object']
        confidence = fused_result['resolution_confidence']

        if target_obj and confidence > 0.6:
            object_desc = f"{target_obj['class']} at ({target_obj['bbox']['center_x']:.0f}, {target_obj['bbox']['center_y']:.0f})"
            responses = {
                'move_forward': f"Moving forward.",
                'move_backward': f"Moving backward.",
                'turn_left': f"Turning left.",
                'turn_right': f"Turning right.",
                'approach_object': f"Approaching the {object_desc}.",
                'avoid_object': f"Avoiding the {object_desc}.",
                'follow_object': f"Following the {object_desc}.",
                'stop': f"Stopping.",
                'pick_up': f"Picking up the {object_desc}.",
                'put_down': f"Putting down object."
            }
        else:
            responses = {
                'move_forward': "Moving forward.",
                'move_backward': "Moving backward.",
                'turn_left': "Turning left.",
                'turn_right': "Turning right.",
                'approach_object': "Approaching object.",
                'avoid_object': "Avoiding object.",
                'follow_object': "Following object.",
                'stop': "Stopping.",
                'pick_up': "Picking up object.",
                'put_down': "Putting down object."
            }

        return responses.get(intent, "Processing command.")

def main(args=None):
    rclpy.init(args=args)
    voice_vision = VoiceVisionAction()

    try:
        rclpy.spin(voice_vision)
    except KeyboardInterrupt:
        voice_vision.get_logger().info('Voice-vision action shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        voice_vision.action_pub.publish(cmd)

        # Stop processing
        voice_vision.is_listening = False

        voice_vision.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Multi-Modal Voice Control

Implementing multi-modal voice control that combines voice, vision, and other sensors:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import AudioData, Image, LaserScan
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import speech_recognition as sr
import threading
import queue
from typing import Dict, List, Optional
import json

class MultiModalVoiceControl(Node):
    def __init__(self):
        super().__init__('multi_modal_voice_control')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

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

        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/multi_modal_status',
            10
        )

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Initialize multi-modal components
        self.modal_fusion_engine = None
        self.initialize_multi_modal_components()

        # State variables
        self.current_audio = None
        self.current_image = None
        self.current_scan = None
        self.current_detections = None
        self.current_command = None
        self.fusion_context = {}

        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = True

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Control timer
        self.multi_modal_timer = self.create_timer(0.1, self.multi_modal_processing_loop)

        self.get_logger().info('Multi-modal voice control initialized')

    def initialize_multi_modal_components(self):
        """Initialize multi-modal processing components"""
        try:
            # Multi-modal fusion engine
            class ModalFusionEngine:
                def __init__(self):
                    self.safety_context = {}
                    self.spatial_context = {}
                    self.object_context = {}
                    self.temporal_context = {}

                def fuse_modalities(self, command: str, vision_data: Dict,
                                  scan_data: Dict, other_sensors: Dict) -> Dict:
                    """Fuse information from multiple modalities"""
                    result = {
                        'command': command,
                        'modalities': {
                            'vision': vision_data,
                            'lidar': scan_data,
                            'other': other_sensors
                        },
                        'fused_intent': '',
                        'action_plan': [],
                        'safety_check': True,
                        'confidence': 0.0,
                        'contextual_adjustments': []
                    }

                    # Perform safety checks using LIDAR data
                    safety_ok, safety_adjustments = self.check_safety(scan_data)
                    result['safety_check'] = safety_ok
                    result['contextual_adjustments'].extend(safety_adjustments)

                    # Interpret command using vision context
                    vision_intent, vision_adjustments = self.interpret_with_vision(
                        command, vision_data
                    )
                    result['fused_intent'] = vision_intent
                    result['contextual_adjustments'].extend(vision_adjustments)

                    # Generate action plan
                    action_plan = self.generate_action_plan(
                        result['fused_intent'],
                        result['modalities'],
                        result['contextual_adjustments']
                    )
                    result['action_plan'] = action_plan

                    # Calculate overall confidence
                    result['confidence'] = self.calculate_confidence(result)

                    return result

                def check_safety(self, scan_data: Dict) -> Tuple[bool, List[str]]:
                    """Check safety using LIDAR data"""
                    adjustments = []
                    safety_ok = True

                    if scan_data and 'ranges' in scan_data:
                        ranges = scan_data['ranges']
                        if ranges:
                            min_distance = min(r for r in ranges if r > 0 and not np.isinf(r))

                            if min_distance < 0.5:  # 50cm safety threshold
                                safety_ok = False
                                adjustments.append(f'OBSTACLE_DETECTED_{min_distance:.2f}M')
                            elif min_distance < 1.0:
                                adjustments.append(f'OBSTACLE_NEAR_{min_distance:.2f}M')

                    return safety_ok, adjustments

                def interpret_with_vision(self, command: str, vision_data: Dict) -> Tuple[str, List[str]]:
                    """Interpret command using visual context"""
                    adjustments = []
                    command_lower = command.lower()

                    # If vision data contains relevant objects, use them for interpretation
                    if vision_data and 'objects' in vision_data:
                        relevant_objects = []
                        for obj in vision_data['objects']:
                            if obj.get('class', '').lower() in command_lower:
                                relevant_objects.append(obj)

                        if relevant_objects:
                            # Command references visible objects
                            if any(word in command_lower for word in ['approach', 'go to', 'toward']):
                                return 'approach_object', [f'TARGET_OBJECT_{obj["class"]}' for obj in relevant_objects]
                            elif any(word in command_lower for word in ['follow', 'chase']):
                                return 'follow_object', [f'TARGET_OBJECT_{obj["class"]}' for obj in relevant_objects]
                            elif any(word in command_lower for word in ['avoid', 'stay away']):
                                return 'avoid_object', [f'AVERSIVE_OBJECT_{obj["class"]}' for obj in relevant_objects]

                    # Default interpretation
                    if any(word in command_lower for word in ['move', 'go', 'forward']):
                        return 'move_forward', []
                    elif any(word in command_lower for word in ['backward', 'back']):
                        return 'move_backward', []
                    elif any(word in command_lower for word in ['turn', 'left']):
                        return 'turn_left', []
                    elif any(word in command_lower for word in ['turn', 'right']):
                        return 'turn_right', []
                    elif any(word in command_lower for word in ['stop', 'halt']):
                        return 'stop', []
                    else:
                        return 'move_forward', []

                def generate_action_plan(self, intent: str, modalities: Dict, adjustments: List[str]) -> List[Dict]:
                    """Generate action plan considering all modalities"""
                    action_plan = []

                    # Base action based on intent
                    base_action = {
                        'intent': intent,
                        'parameters': {},
                        'priority': 1
                    }

                    # Apply safety adjustments
                    for adjustment in adjustments:
                        if adjustment.startswith('OBSTACLE_DETECTED'):
                            # Override with safety action
                            safety_action = {
                                'intent': 'avoid_obstacle',
                                'parameters': {'safety_distance': 0.5},
                                'priority': 10
                            }
                            action_plan.append(safety_action)
                        elif adjustment.startswith('OBSTACLE_NEAR'):
                            # Modify base action
                            base_action['parameters']['speed_reduction'] = True

                    # Add base action
                    action_plan.append(base_action)

                    return action_plan

                def calculate_confidence(self, result: Dict) -> float:
                    """Calculate overall confidence in interpretation"""
                    # Base confidence
                    confidence = 0.5

                    # Boost confidence if multiple modalities agree
                    if result['modalities']['vision']:
                        confidence += 0.2
                    if result['modalities']['lidar']:
                        confidence += 0.2

                    # Reduce confidence if safety issues detected
                    if not result['safety_check']:
                        confidence -= 0.3

                    # Ensure confidence is between 0 and 1
                    return max(0.0, min(1.0, confidence))

            self.modal_fusion_engine = ModalFusionEngine()
            self.get_logger().info('Multi-modal fusion engine initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize multi-modal components: {str(e)}')

    def audio_callback(self, msg):
        """Process audio data"""
        self.current_audio = msg

    def image_callback(self, msg):
        """Process camera image"""
        self.current_image = msg

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Convert scan message to dictionary format
        scan_dict = {
            'ranges': np.array(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        }
        self.current_scan = scan_dict

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def audio_processing_loop(self):
        """Background audio processing loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=5)

                self.audio_queue.put(audio)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {str(e)}')
                continue

    def multi_modal_processing_loop(self):
        """Main multi-modal processing loop"""
        try:
            # Process new audio if available
            while not self.audio_queue.empty():
                audio = self.audio_queue.get()

                # Convert speech to text
                command_text = self.speech_to_text(audio)

                if command_text:
                    # Process command with all modalities
                    self.process_multi_modal_command(command_text)

        except Exception as e:
            self.get_logger().error(f'Error in multi-modal processing: {str(e)}')

    def speech_to_text(self, audio):
        """Convert speech to text"""
        try:
            text = self.recognizer.recognize_google(audio)
            self.get_logger().info(f'Recognized: "{text}"')
            return text.lower()
        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio')
            return None
        except sr.RequestError as e:
            self.get_logger().error(f'Speech recognition error: {str(e)}')
            return None

    def process_multi_modal_command(self, command_text: str):
        """Process command using all available modalities"""
        # Prepare data from all modalities
        vision_data = self.prepare_vision_data()
        scan_data = self.current_scan if self.current_scan else {}
        other_sensors = {}  # Could include IMU, odometry, etc.

        # Fuse all modalities
        fusion_result = self.modal_fusion_engine.fuse_modalities(
            command_text, vision_data, scan_data, other_sensors
        )

        # Check safety before executing
        if not fusion_result['safety_check']:
            self.get_logger().warn('Safety check failed, not executing command')

            # Publish safety warning
            status_msg = String()
            status_msg.data = 'SAFETY_HAZARD_DETECTED'
            self.status_pub.publish(status_msg)
            return

        # Execute action plan
        for action in fusion_result['action_plan']:
            if action['priority'] >= 10:  # Safety-critical actions
                action_cmd = self.execute_safety_action(action)
            else:
                action_cmd = self.execute_normal_action(action)

            if action_cmd is not None:
                self.action_pub.publish(action_cmd)

        # Publish command
        cmd_msg = String()
        cmd_msg.data = command_text
        self.command_pub.publish(cmd_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f'Command: "{command_text}", Intent: {fusion_result["fused_intent"]}, Confidence: {fusion_result["confidence"]:.2f}'
        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f'Multi-Modal Command: "{command_text}", '
            f'Intent: {fusion_result["fused_intent"]}, '
            f'Confidence: {fusion_result["confidence"]:.2f}, '
            f'Safety: {"OK" if fusion_result["safety_check"] else "FAILED"}'
        )

    def prepare_vision_data(self) -> Dict:
        """Prepare vision data for multi-modal fusion"""
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
                    'attributes': []
                }

                vision_data['objects'].append(obj_data)

        return vision_data

    def execute_safety_action(self, action: Dict) -> Twist:
        """Execute safety-critical action"""
        cmd = Twist()

        if action['intent'] == 'avoid_obstacle':
            # Emergency stop or obstacle avoidance
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn to avoid
        else:
            # Default stop for safety
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def execute_normal_action(self, action: Dict) -> Twist:
        """Execute normal action"""
        intent = action['intent']
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
        elif intent == 'avoid_object':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
        elif intent == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif intent == 'follow_object':
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Default: move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        # Apply any parameters
        params = action.get('parameters', {})
        if params.get('speed_reduction'):
            cmd.linear.x *= 0.5  # Reduce speed
            cmd.angular.z *= 0.5

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    multi_modal = MultiModalVoiceControl()

    try:
        rclpy.spin(multi_modal)
    except KeyboardInterrupt:
        multi_modal.get_logger().info('Multi-modal voice control shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        multi_modal.action_pub.publish(cmd)

        # Stop processing
        multi_modal.is_listening = False

        multi_modal.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Feedback and Confirmation

Implementing feedback and confirmation mechanisms for voice commands:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
import numpy as np
import speech_recognition as sr
import threading
import queue
from typing import Dict, List
import time

class VoiceCommandFeedback(Node):
    def __init__(self):
        super().__init__('voice_command_feedback')

        # Publishers and subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.confirmation_pub = self.create_publisher(
            String,
            '/command_confirmation',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            '/command_feedback',
            10
        )

        self.status_pub = self.create_publisher(
            Bool,
            '/command_acknowledged',
            10
        )

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Initialize feedback components
        self.command_history = []
        self.pending_confirmation = None
        self.confirmation_timeout = 5.0  # seconds
        self.last_confirmation_time = None

        # State variables
        self.current_command = None
        self.command_queue = queue.Queue()
        self.feedback_enabled = True

        # Control timers
        self.feedback_timer = self.create_timer(0.1, self.feedback_processing_loop)
        self.confirmation_timer = self.create_timer(1.0, self.confirmation_check_loop)

        self.get_logger().info('Voice command feedback system initialized')

    def audio_callback(self, msg):
        """Process audio data"""
        # This node focuses on feedback rather than audio processing
        pass

    def command_callback(self, msg):
        """Process incoming voice command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command for feedback: "{self.current_command}"')

        # Add to command queue for processing
        self.command_queue.put({
            'command': self.current_command,
            'timestamp': time.time(),
            'attempts': 0
        })

    def feedback_processing_loop(self):
        """Main feedback processing loop"""
        try:
            # Process new commands
            while not self.command_queue.empty():
                command_info = self.command_queue.get()

                # Process the command with feedback
                self.process_command_with_feedback(command_info)

        except Exception as e:
            self.get_logger().error(f'Error in feedback processing: {str(e)}')

    def process_command_with_feedback(self, command_info: Dict):
        """Process command with feedback and confirmation"""
        command = command_info['command']

        # Generate feedback
        feedback = self.generate_feedback(command)

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = feedback
        self.feedback_pub.publish(feedback_msg)

        # Add to history
        self.command_history.append({
            'command': command,
            'feedback': feedback,
            'timestamp': command_info['timestamp'],
            'confirmed': False,
            'attempts': command_info['attempts']
        })

        # Request confirmation for complex commands
        if self.requires_confirmation(command):
            self.request_confirmation(command)
        else:
            # For simple commands, just acknowledge
            self.acknowledge_command(command)

        self.get_logger().info(f'Feedback provided: "{feedback}" for command: "{command}"')

    def generate_feedback(self, command: str) -> str:
        """Generate appropriate feedback for the command"""
        command_lower = command.lower()

        # Map commands to feedback responses
        if any(word in command_lower for word in ['move', 'go', 'forward']):
            return f"Okay, moving forward."
        elif any(word in command_lower for word in ['backward', 'back']):
            return f"Okay, moving backward."
        elif any(word in command_lower for word in ['turn', 'left']):
            return f"Okay, turning left."
        elif any(word in command_lower for word in ['turn', 'right']):
            return f"Okay, turning right."
        elif any(word in command_lower for word in ['approach', 'go to', 'toward']):
            return f"Okay, approaching as requested."
        elif any(word in command_lower for word in ['avoid', 'stay away']):
            return f"Okay, avoiding obstacles."
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return f"Okay, stopping."
        elif any(word in command_lower for word in ['follow', 'chase', 'track']):
            return f"Okay, following."
        else:
            return f"Okay, executing command: {command}"

    def requires_confirmation(self, command: str) -> bool:
        """Determine if command requires explicit confirmation"""
        command_lower = command.lower()

        # Commands that require confirmation
        confirmation_required = [
            'pick up', 'grab', 'take', 'lift', 'collect', 'get',
            'put down', 'drop', 'place', 'set down', 'release',
            'go to', 'move to', 'approach', 'come to',
            'follow', 'chase', 'track'
        ]

        for req in confirmation_required:
            if req in command_lower:
                return True

        return False

    def request_confirmation(self, command: str):
        """Request explicit confirmation for the command"""
        confirmation_msg = String()
        confirmation_msg.data = f'Did you say "{command}"? Please confirm with yes or no.'
        self.confirmation_pub.publish(confirmation_msg)

        # Store pending confirmation
        self.pending_confirmation = {
            'command': command,
            'timestamp': time.time(),
            'attempts': 0
        }

        self.get_logger().info(f'Confirmation requested: "{confirmation_msg.data}"')

    def acknowledge_command(self, command: str):
        """Acknowledge that command was understood and will be executed"""
        ack_msg = Bool()
        ack_msg.data = True
        self.status_pub.publish(ack_msg)

        self.get_logger().info(f'Command acknowledged: "{command}"')

    def confirmation_check_loop(self):
        """Check for pending confirmations and handle timeouts"""
        if self.pending_confirmation:
            elapsed = time.time() - self.pending_confirmation['timestamp']

            if elapsed > self.confirmation_timeout:
                # Confirmation timeout - ask again or execute anyway
                self.handle_confirmation_timeout()

    def handle_confirmation_timeout(self):
        """Handle confirmation timeout"""
        if self.pending_confirmation:
            command = self.pending_confirmation['command']
            attempts = self.pending_confirmation['attempts']

            if attempts < 2:  # Retry once
                self.get_logger().info(f'Confirmation timeout for: "{command}", retrying...')

                # Retry confirmation
                self.pending_confirmation['timestamp'] = time.time()
                self.pending_confirmation['attempts'] += 1

                retry_msg = String()
                retry_msg.data = f'I did not receive confirmation for "{command}". Executing anyway.'
                self.confirmation_pub.publish(retry_msg)

                # Acknowledge and execute
                self.acknowledge_command(command)
            else:
                # Give up on confirmation
                self.get_logger().info(f'Giving up on confirmation for: "{command}"')

                # Acknowledge anyway
                self.acknowledge_command(command)

            # Clear pending confirmation
            self.pending_confirmation = None

    def confirmation_response_callback(self, msg):
        """Process confirmation response (would be from speech recognition of 'yes'/'no')"""
        response = msg.data.lower().strip()

        if self.pending_confirmation:
            command = self.pending_confirmation['command']

            if response in ['yes', 'yep', 'yeah', 'confirm', 'correct', 'right']:
                # Confirmation received
                self.get_logger().info(f'Confirmation received for: "{command}"')
                self.acknowledge_command(command)
                self.pending_confirmation = None

            elif response in ['no', 'nope', 'wrong', 'cancel', 'stop']:
                # Negative confirmation - cancel command
                self.get_logger().info(f'Command rejected: "{command}"')

                reject_msg = String()
                reject_msg.data = f'Command "{command}" has been cancelled.'
                self.feedback_pub.publish(reject_msg)

                self.pending_confirmation = None

    def get_recent_commands(self, limit: int = 5) -> List[Dict]:
        """Get recent commands from history"""
        return self.command_history[-limit:] if len(self.command_history) >= limit else self.command_history

def main(args=None):
    rclpy.init(args=args)
    feedback_system = VoiceCommandFeedback()

    try:
        rclpy.spin(feedback_system)
    except KeyboardInterrupt:
        feedback_system.get_logger().info('Voice command feedback system shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        feedback_system.action_pub.publish(cmd)

        feedback_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Voice-to-Action Pipelines

1. **Robust Speech Recognition**: Handle various accents, speaking styles, and environmental conditions
2. **Context Awareness**: Maintain conversation and spatial context for better understanding
3. **Safety First**: Implement safety checks before executing potentially dangerous commands
4. **Feedback Mechanisms**: Provide clear feedback to users about command recognition and execution
5. **Error Recovery**: Handle recognition errors gracefully with clarification requests
6. **Latency Optimization**: Minimize processing delay for natural interaction
7. **Privacy Considerations**: Protect user privacy in speech processing
8. **Multi-Modal Integration**: Combine voice with vision and other sensors for better understanding

### Physical Grounding and Simulation-to-Real Mapping

When implementing voice-to-action pipelines:

- **Acoustic Environment**: Account for real-world noise and reverberation
- **Microphone Quality**: Consider microphone placement and quality in real robots
- **Processing Delays**: Account for computational delays in real-time systems
- **Network Dependencies**: Consider connectivity requirements for cloud-based recognition
- **Safety Systems**: Implement safety checks around voice-controlled actions
- **User Adaptation**: Allow for user-specific speech patterns and preferences
- **Environmental Factors**: Consider acoustic properties of different environments

### Troubleshooting Voice-to-Action Issues

Common voice-to-action problems and solutions:

- **Recognition Errors**: Improve acoustic models and noise reduction
- **Context Loss**: Implement better context tracking and recovery
- **Ambiguity**: Develop disambiguation strategies and user clarification
- **Performance Issues**: Optimize models and consider edge processing
- **Integration Problems**: Ensure proper data flow between components
- **Safety Issues**: Implement comprehensive safety checks and overrides

### Summary

This chapter covered voice-to-action pipelines that convert spoken language commands into robotic actions. You learned about basic and advanced voice-to-action implementations, multi-modal integration with vision and other sensors, and feedback mechanisms for natural human-robot interaction. Voice-to-action pipelines enable intuitive human-robot interfaces, making robots more accessible and usable in real-world applications. The integration of voice with other modalities enhances the robot's ability to understand and execute commands in complex, dynamic environments. In the next chapter, we'll explore cognitive robotics and planning for intelligent robot behavior.