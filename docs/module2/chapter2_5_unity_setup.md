# Module 2: Digital Twins – Gazebo & Unity

## Chapter 2.5: Unity 3D Simulation Environment Setup

This chapter introduces Unity 3D as a simulation environment for robotics applications, focusing on setting up Unity robotics packages and creating simulation environments for robotic systems.

### Introduction to Unity for Robotics

Unity is a powerful 3D development platform that has gained significant traction in robotics simulation due to its high-quality rendering capabilities, flexible physics engine, and extensive asset ecosystem. Unity's robotics packages provide integration with ROS/ROS2, enabling seamless communication between Unity simulations and robotic systems.

### Unity Robotics Packages Overview

Unity provides several packages specifically designed for robotics development:

1. **Unity Robotics Hub**: Centralized installer for robotics packages
2. **Unity Robotics Package (URP)**: Core robotics functionality
3. **ROS-TCP-Connector**: Communication bridge between Unity and ROS/ROS2
4. **Unity Perception Package**: Tools for generating synthetic training data
5. **Unity ML-Agents**: Machine learning framework for robotics applications

### Installing Unity and Robotics Packages

#### Prerequisites
- Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- Unity Hub (recommended)
- Unity 2021.3 LTS or later
- ROS 2 Humble Hawksbill (for ROS 2 integration)

#### Installing Unity Hub and Editor

1. Download Unity Hub from the Unity website
2. Install Unity Hub and create an account
3. Use Unity Hub to install Unity Editor 2021.3 LTS or later

#### Installing Robotics Packages via Unity Robotics Hub

```bash
# Install Unity Robotics Hub from GitHub
git clone https://github.com/Unity-Technologies/Unity-Robotics-Hub.git
cd Unity-Robotics-Hub
```

The Unity Robotics Hub provides a GUI for installing and managing robotics packages.

#### Manual Package Installation

Alternatively, install packages directly in Unity:

1. Open Unity Editor
2. Go to Window > Package Manager
3. Click the + button > Add package from git URL
4. Add the following packages:
   - `com.unity.robotics.ros-tcp-connector`
   - `com.unity.perception`
   - `com.unity.ml-agents`

### Setting Up a Basic Robotics Scene

#### Creating a New Unity Project

1. Open Unity Hub
2. Create a new 3D project
3. Name it "RoboticsSimulation"
4. Select the project folder

#### Basic Scene Setup

Here's a basic C# script to initialize a ROS connection in Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    [SerializeField]
    private string rosIPAddress = "127.0.0.1";
    [SerializeField]
    private int rosPort = 10000;

    private ROSConnection ros;

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.instance;

        // Set the IP address and port for the ROS connection
        ros.Initialize(rosIPAddress, rosPort);

        Debug.Log("ROS connection initialized");
    }

    void Update()
    {
        // Update loop for continuous operations
    }
}
```

### ROS-TCP-Connector Integration

The ROS-TCP-Connector package enables communication between Unity and ROS 2. Here's how to set it up:

#### Basic ROS Publisher in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class UnityPublisher : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/unity_data";
    [SerializeField]
    private float publishFrequency = 10f; // Hz

    private ROSConnection ros;
    private float publishTimer;

    void Start()
    {
        ros = ROSConnection.instance;
        publishTimer = 1f / publishFrequency;
    }

    void Update()
    {
        publishTimer -= Time.deltaTime;
        if (publishTimer <= 0)
        {
            // Create and publish a message
            ros.Publish(topicName, new StringMsg { data = "Hello from Unity! " + Time.time });
            publishTimer = 1f / publishFrequency;
        }
    }
}
```

#### Basic ROS Subscriber in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class UnitySubscriber : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/robot_cmd";

    void Start()
    {
        // Subscribe to a ROS topic
        ROSConnection.instance.Subscribe<StringMsg>(topicName, OnMessageReceived);
    }

    void OnMessageReceived(StringMsg message)
    {
        Debug.Log("Received message: " + message.data);

        // Process the received command
        ProcessCommand(message.data);
    }

    void ProcessCommand(string command)
    {
        // Handle the command received from ROS
        switch (command)
        {
            case "move_forward":
                MoveRobot(Vector3.forward);
                break;
            case "turn_left":
                RotateRobot(-90f);
                break;
            case "turn_right":
                RotateRobot(90f);
                break;
        }
    }

    void MoveRobot(Vector3 direction)
    {
        transform.Translate(direction * Time.deltaTime);
    }

    void RotateRobot(float angle)
    {
        transform.Rotate(Vector3.up, angle * Time.deltaTime);
    }
}
```

### Creating a Simple Robot Model

Here's an example of how to create a simple robot model in Unity:

```csharp
using UnityEngine;

public class SimpleRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float linearSpeed = 1.0f;
    public float angularSpeed = 50.0f;

    [Header("Wheel Configuration")]
    public Transform leftWheel;
    public Transform rightWheel;
    public float wheelRadius = 0.1f;

    [Header("ROS Communication")]
    public string cmdVelTopic = "/cmd_vel";
    public string odomTopic = "/odom";

    private float leftWheelVelocity = 0f;
    private float rightWheelVelocity = 0f;

    // For differential drive kinematics
    private float robotLinearVelocity = 0f;
    private float robotAngularVelocity = 0f;

    void Start()
    {
        // Initialize ROS connection
        ROSConnection.instance.Subscribe<geometry_msgs.Twist>(cmdVelTopic, OnCmdVelReceived);
    }

    void OnCmdVelReceived(geometry_msgs.Twist cmdVel)
    {
        // Convert Twist message to wheel velocities for differential drive
        float linear = cmdVel.linear.x;
        float angular = cmdVel.angular.z;

        // Calculate wheel velocities for differential drive
        float wheelSeparation = Vector3.Distance(
            leftWheel.position,
            rightWheel.position
        );

        leftWheelVelocity = (linear - angular * wheelSeparation / 2.0f) / wheelRadius;
        rightWheelVelocity = (linear + angular * wheelSeparation / 2.0f) / wheelRadius;

        // Store robot velocities for odometry
        robotLinearVelocity = linear;
        robotAngularVelocity = angular;
    }

    void Update()
    {
        // Update wheel rotations based on velocities
        if (leftWheel != null)
        {
            leftWheel.Rotate(Vector3.right, leftWheelVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }

        if (rightWheel != null)
        {
            rightWheel.Rotate(Vector3.right, rightWheelVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }

        // Update robot position based on velocities
        transform.Translate(Vector3.forward * robotLinearVelocity * Time.deltaTime);
        transform.Rotate(Vector3.up, robotAngularVelocity * Time.deltaTime * Mathf.Rad2Deg);

        // Publish odometry (simplified)
        PublishOdometry();
    }

    void PublishOdometry()
    {
        // Create and publish odometry message
        var odomMsg = new nav_msgs.Odometry
        {
            header = new std_msgs.Header { frame_id = "odom" },
            child_frame_id = "base_link",
            pose = new geometry_msgs.PoseWithCovariance
            {
                pose = new geometry_msgs.Pose
                {
                    position = new geometry_msgs.Point
                    {
                        x = transform.position.x,
                        y = transform.position.y,
                        z = transform.position.z
                    },
                    orientation = new geometry_msgs.Quaternion
                    {
                        x = transform.rotation.x,
                        y = transform.rotation.y,
                        z = transform.rotation.z,
                        w = transform.rotation.w
                    }
                }
            },
            twist = new geometry_msgs.TwistWithCovariance
            {
                twist = new geometry_msgs.Twist
                {
                    linear = new geometry_msgs.Vector3
                    {
                        x = robotLinearVelocity,
                        y = 0,
                        z = 0
                    },
                    angular = new geometry_msgs.Vector3
                    {
                        x = 0,
                        y = 0,
                        z = robotAngularVelocity
                    }
                }
            }
        };

        ROSConnection.instance.Publish(odomTopic, odomMsg);
    }
}
```

### Setting Up Unity Perception Package

The Unity Perception package enables synthetic data generation for training AI models:

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.Labeling;

public class PerceptionSetup : MonoBehaviour
{
    [Header("Perception Configuration")]
    public bool enableGroundTruth = true;
    public float captureFrequency = 0.1f; // seconds between captures

    private float captureTimer;

    void Start()
    {
        if (enableGroundTruth)
        {
            // Enable perception system
            PerceptionSettings.Initialize();

            // Set capture frequency
            PerceptionSettings.CaptureEveryNthFrame = Mathf.RoundToInt(1f / (captureFrequency * Time.fixedDeltaTime));
        }
    }

    void Update()
    {
        if (enableGroundTruth)
        {
            captureTimer += Time.deltaTime;
            if (captureTimer >= captureFrequency)
            {
                // Trigger perception capture
                PerceptionSettings.SynchronousRendering = true;
                captureTimer = 0f;
            }
        }
    }

    void OnValidate()
    {
        // Ensure capture frequency is positive
        captureFrequency = Mathf.Max(0.01f, captureFrequency);
    }
}
```

### Unity ML-Agents Integration

For reinforcement learning applications, Unity ML-Agents can be integrated:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class RobotAgent : Agent
{
    [Header("Robot Configuration")]
    public Transform target;
    public float moveSpeed = 1.0f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position
        transform.position = new Vector3(Random.Range(-5f, 5f), 0.5f, Random.Range(-5f, 5f));

        // Reset target position
        target.position = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe relative position to target
        sensor.AddObservation((target.position - transform.position).normalized);

        // Observe robot velocity
        sensor.AddObservation(rb.velocity);

        // Observe robot rotation
        sensor.AddObservation(transform.rotation);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Get actions
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];

        // Apply movement
        rb.AddForce(controlSignal * moveSpeed);

        // Simple reward system
        float distanceToTarget = Vector3.Distance(transform.position, target.position);

        // Give reward for getting closer to target
        SetReward(1.0f / (1.0f + distanceToTarget));

        // End episode if close enough to target
        if (distanceToTarget < 1.0f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // End episode if too far away
        if (distanceToTarget > 20.0f)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

### Unity Scene Setup for Robotics

Creating a proper scene setup for robotics simulation:

1. **Environment Setup**:
   - Add lighting (preferably HDRI for realistic lighting)
   - Create ground plane with appropriate physics materials
   - Add obstacles and environment elements

2. **Physics Configuration**:
   - Configure Physics settings for realistic simulation
   - Set appropriate gravity (usually -9.81 on Y-axis)
   - Configure collision layers if needed

3. **Camera Setup**:
   - Add main camera for visualization
   - Consider adding multiple cameras for different sensor views
   - Configure camera properties for sensor simulation

### ROS Bridge Configuration

To connect Unity with ROS 2, you need to set up the ROS bridge:

```bash
# Terminal 1: Start ROS 2 daemon
source /opt/ros/humble/setup.bash
ros2 daemon start

# Terminal 2: Start the ROS TCP endpoint
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000
```

### Best Practices for Unity Robotics Development

1. **Performance Optimization**:
   - Use appropriate polygon counts for real-time simulation
   - Implement Level of Detail (LOD) for complex models
   - Use occlusion culling for large environments
   - Optimize physics calculations

2. **Realistic Simulation**:
   - Match physics properties to real-world values
   - Implement appropriate sensor noise models
   - Consider environmental factors (lighting, weather)

3. **Modular Design**:
   - Create reusable components for different robot types
   - Implement proper scene management
   - Use ScriptableObjects for configuration

4. **Testing and Validation**:
   - Validate simulation behavior against real hardware when possible
   - Implement proper logging and debugging tools
   - Create test scenarios to verify functionality

### Physical Grounding and Simulation-to-Real Mapping

When developing Unity simulations for robotics:

- **Physics Properties**: Ensure material properties match real-world values
- **Sensor Models**: Implement realistic sensor noise and limitations
- **Timing**: Consider real-time constraints and communication delays
- **Environmental Conditions**: Account for lighting, weather, and other environmental factors
- **Hardware Limitations**: Model computational and power constraints of real hardware

### Troubleshooting Common Issues

1. **Connection Issues**:
   - Verify IP addresses and ports match between Unity and ROS
   - Check firewall settings
   - Ensure ROS TCP endpoint is running

2. **Performance Issues**:
   - Reduce scene complexity
   - Optimize rendering settings
   - Use appropriate time scaling

3. **Physics Issues**:
   - Verify mass and collision properties
   - Adjust physics timestep if needed
   - Check for overlapping colliders

### Summary

This chapter introduced Unity 3D as a simulation environment for robotics applications. You learned how to set up Unity robotics packages, create basic robot controllers, integrate with ROS 2, and implement perception and learning capabilities. Unity provides a powerful platform for creating realistic simulation environments with high-quality rendering and flexible physics. In the next chapter, we'll explore testing and validation in simulated environments before hardware deployment.