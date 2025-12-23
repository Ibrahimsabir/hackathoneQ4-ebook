# Module 3: AI Robot Brain – NVIDIA Isaac

## Chapter 3.4: Control Algorithms with Isaac

This chapter explores control algorithms using NVIDIA Isaac ROS, focusing on how to implement optimized control systems that leverage Isaac's hardware acceleration and NVIDIA's computing capabilities for precise robot motion control.

### Understanding Isaac ROS Control Systems

Isaac ROS provides optimized control algorithms that take advantage of NVIDIA's hardware acceleration for real-time robotics applications. The key control components include:

- **PID Controllers**: Proportional-Integral-Derivative controllers for precise control
- **MPC Controllers**: Model Predictive Control for advanced trajectory following
- **Adaptive Control**: Controllers that adjust parameters based on system behavior
- **Optimal Control**: Algorithms that minimize control effort while achieving goals

### Isaac ROS Control Architecture

The Isaac ROS control system architecture includes:

```
+-------------------+
|   Trajectory      |
|   Generator      |
+-------------------+
|   Controller      |
|   (PID/MPC)      |
+-------------------+
|   Hardware        |
|   Interface      |
+-------------------+
|   Robot           |
|   Dynamics       |
+-------------------+
```

### PID Control with Isaac ROS

Implementing a PID controller optimized for Isaac ROS:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import time

class IsaacPIDController(Node):
    def __init__(self):
        super().__init__('isaac_pid_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_callback,
            10
        )

        # PID controller parameters
        self.kp_linear = 1.0  # Proportional gain for linear velocity
        self.ki_linear = 0.1  # Integral gain for linear velocity
        self.kd_linear = 0.05  # Derivative gain for linear velocity

        self.kp_angular = 2.0  # Proportional gain for angular velocity
        self.ki_angular = 0.2  # Integral gain for angular velocity
        self.kd_angular = 0.1  # Derivative gain for angular velocity

        # PID state variables
        self.linear_error_sum = 0.0
        self.linear_error_prev = 0.0
        self.angular_error_sum = 0.0
        self.angular_error_prev = 0.0

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.target_pose = None

        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        # Time tracking for PID
        self.prev_time = time.time()

        self.get_logger().info('Isaac PID controller initialized')

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def target_callback(self, msg):
        """Process target pose"""
        self.target_pose = msg.pose

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or self.target_pose is None:
            return

        # Calculate current time and time delta
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0:
            return

        # Calculate errors
        linear_error = self.calculate_linear_error()
        angular_error = self.calculate_angular_error()

        # Update PID calculations
        linear_cmd = self.calculate_pid_command(
            linear_error, dt, self.linear_error_sum, self.linear_error_prev,
            self.kp_linear, self.ki_linear, self.kd_linear
        )
        self.linear_error_sum = linear_cmd[1]
        self.linear_error_prev = linear_error

        angular_cmd = self.calculate_pid_command(
            angular_error, dt, self.angular_error_sum, self.angular_error_prev,
            self.kp_angular, self.ki_angular, self.kd_angular
        )
        self.angular_error_sum = angular_cmd[1]
        self.angular_error_prev = angular_error

        # Create and publish velocity command
        cmd = Twist()
        cmd.linear.x = linear_cmd[0]
        cmd.angular.z = angular_cmd[0]

        # Limit velocities to safe values
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        self.cmd_vel_pub.publish(cmd)

        # Log control information
        self.get_logger().info(
            f'PID Control - Linear: {cmd.linear.x:.2f}, '
            f'Angular: {cmd.angular.z:.2f}, '
            f'Linear Error: {linear_error:.2f}, '
            f'Angular Error: {angular_error:.2f}'
        )

    def calculate_linear_error(self):
        """Calculate linear distance error to target"""
        if self.current_pose is None or self.target_pose is None:
            return 0.0

        dx = self.target_pose.position.x - self.current_pose.position.x
        dy = self.target_pose.position.y - self.current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        return distance

    def calculate_angular_error(self):
        """Calculate angular error to target"""
        if self.current_pose is None or self.target_pose is None:
            return 0.0

        # Calculate desired angle to target
        dx = self.target_pose.position.x - self.current_pose.position.x
        dy = self.target_pose.position.y - self.current_pose.position.y
        desired_angle = np.arctan2(dy, dx)

        # Get current orientation (simplified from quaternion)
        # In a real implementation, you'd convert quaternion to yaw
        current_angle = 0.0  # Simplified

        # Calculate angle error
        angle_error = desired_angle - current_angle

        # Normalize angle error
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        return angle_error

    def calculate_pid_command(self, error, dt, error_sum, error_prev, kp, ki, kd):
        """Calculate PID command"""
        # Proportional term
        p_term = kp * error

        # Integral term
        error_sum += error * dt
        i_term = ki * error_sum

        # Derivative term
        if dt > 0:
            d_term = kd * (error - error_prev) / dt
        else:
            d_term = 0.0

        # Total PID command
        command = p_term + i_term + d_term

        return command, error_sum

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacPIDController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('PID controller shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        controller.cmd_vel_pub.publish(cmd)

        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Model Predictive Control (MPC)

Implementing an MPC controller for advanced trajectory following:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
import numpy as np
import cvxpy as cp  # Requires: pip install cvxpy

class IsaacMPCController(Node):
    def __init__(self):
        super().__init__('isaac_mpc_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(
            Path,
            '/trajectory',
            self.path_callback,
            10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.pose_callback,
            10
        )

        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1      # Time step
        self.n_states = 4  # [x, y, vx, vy]
        self.n_controls = 2  # [ax, ay] - acceleration

        # MPC weights
        self.state_weight = np.diag([10.0, 10.0, 1.0, 1.0])  # Q matrix
        self.control_weight = np.diag([1.0, 1.0])  # R matrix
        self.terminal_weight = np.diag([50.0, 50.0, 5.0, 5.0])  # P matrix

        # Robot state
        self.current_state = np.zeros(self.n_states)
        self.trajectory = []
        self.trajectory_index = 0

        # Control timer
        self.control_timer = self.create_timer(0.05, self.mpc_control_loop)

        self.get_logger().info('Isaac MPC controller initialized')

    def path_callback(self, msg):
        """Process trajectory path"""
        self.trajectory = []
        for pose in msg.poses:
            state = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                0.0,  # vx (estimated)
                0.0   # vy (estimated)
            ])
            self.trajectory.append(state)

        self.trajectory_index = 0

    def pose_callback(self, msg):
        """Process current pose"""
        self.current_state[0] = msg.pose.position.x
        self.current_state[1] = msg.pose.position.y

        # In a real implementation, you'd also get velocity from odometry

    def mpc_control_loop(self):
        """Main MPC control loop"""
        if len(self.trajectory) == 0:
            return

        # Get reference trajectory for the horizon
        ref_trajectory = self.get_reference_trajectory()

        # Solve MPC optimization problem
        control_sequence = self.solve_mpc(ref_trajectory)

        if control_sequence is not None and len(control_sequence) > 0:
            # Apply first control command
            cmd = Twist()
            cmd.linear.x = control_sequence[0][0]  # Simplified mapping
            cmd.angular.z = control_sequence[0][1]  # Simplified mapping

            # Limit commands
            cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

            self.cmd_vel_pub.publish(cmd)

            self.get_logger().info(f'MPC Command - Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}')

    def get_reference_trajectory(self):
        """Get reference trajectory for the prediction horizon"""
        ref_traj = []
        start_idx = self.trajectory_index

        for i in range(self.horizon):
            idx = start_idx + i
            if idx < len(self.trajectory):
                ref_traj.append(self.trajectory[idx])
            else:
                # Use the last point if trajectory is shorter
                ref_traj.append(self.trajectory[-1])

        return ref_traj

    def solve_mpc(self, ref_trajectory):
        """Solve the MPC optimization problem"""
        try:
            # Define optimization variables
            states = cp.Variable((self.horizon + 1, self.n_states))
            controls = cp.Variable((self.horizon, self.n_controls))

            # Objective function
            objective = 0

            # Add state and control costs
            for k in range(self.horizon):
                state_error = states[k] - ref_trajectory[k]
                objective += cp.quad_form(state_error, self.state_weight)
                objective += cp.quad_form(controls[k], self.control_weight)

            # Add terminal cost
            terminal_error = states[self.horizon] - ref_trajectory[-1]
            objective += cp.quad_form(terminal_error, self.terminal_weight)

            # Constraints
            constraints = []

            # Initial state constraint
            constraints.append(states[0] == self.current_state)

            # Dynamics constraints (simplified linear dynamics)
            for k in range(self.horizon):
                # x[k+1] = A*x[k] + B*u[k]
                # For a simple point mass model:
                # x_{k+1} = x_k + v_k * dt
                # v_{k+1} = v_k + a_k * dt
                A = np.array([
                    [1, 0, self.dt, 0],
                    [0, 1, 0, self.dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                B = np.array([
                    [0.5 * self.dt**2, 0],
                    [0, 0.5 * self.dt**2],
                    [self.dt, 0],
                    [0, self.dt]
                ])

                next_state = A @ states[k] + B @ controls[k]
                constraints.append(states[k+1] == next_state)

            # Control constraints (limits)
            for k in range(self.horizon):
                constraints.append(cp.norm(controls[k], 'inf') <= 1.0)

            # Create and solve optimization problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                # Extract optimal control sequence
                optimal_controls = [controls[k].value for k in range(self.horizon)]
                return optimal_controls
            else:
                self.get_logger().warn('MPC optimization failed')
                return None

        except Exception as e:
            self.get_logger().error(f'MPC optimization error: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacMPCController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('MPC controller shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        controller.cmd_vel_pub.publish(cmd)

        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Adaptive Control

Implementing an adaptive control system that adjusts parameters based on performance:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import time

class IsaacAdaptiveController(Node):
    def __init__(self):
        super().__init__('isaac_adaptive_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_callback,
            10
        )

        # Initial PID parameters
        self.kp_linear = 1.0
        self.ki_linear = 0.1
        self.kd_linear = 0.05
        self.kp_angular = 2.0
        self.ki_angular = 0.2
        self.kd_angular = 0.1

        # Adaptive control parameters
        self.adaptation_rate = 0.001
        self.performance_threshold = 0.1
        self.max_adaptation = 0.5

        # State variables
        self.current_pose = None
        self.target_pose = None
        self.linear_error_history = []
        self.angular_error_history = []
        self.max_history = 50

        # PID state variables
        self.linear_error_sum = 0.0
        self.linear_error_prev = 0.0
        self.angular_error_sum = 0.0
        self.angular_error_prev = 0.0

        # Time tracking
        self.prev_time = time.time()

        # Control timer
        self.control_timer = self.create_timer(0.05, self.adaptive_control_loop)

        self.get_logger().info('Isaac adaptive controller initialized')

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose

    def target_callback(self, msg):
        """Process target pose"""
        self.target_pose = msg.pose

    def adaptive_control_loop(self):
        """Main adaptive control loop"""
        if self.current_pose is None or self.target_pose is None:
            return

        # Calculate current time and time delta
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0:
            return

        # Calculate errors
        linear_error = self.calculate_linear_error()
        angular_error = self.calculate_angular_error()

        # Store error history
        self.linear_error_history.append(abs(linear_error))
        self.angular_error_history.append(abs(angular_error))

        # Keep only recent history
        if len(self.linear_error_history) > self.max_history:
            self.linear_error_history.pop(0)
        if len(self.angular_error_history) > self.max_history:
            self.angular_error_history.pop(0)

        # Calculate performance metrics
        avg_linear_error = np.mean(self.linear_error_history) if self.linear_error_history else 0
        avg_angular_error = np.mean(self.angular_error_history) if self.angular_error_history else 0

        # Adapt controller parameters based on performance
        self.adapt_parameters(avg_linear_error, avg_angular_error)

        # Calculate PID commands with adapted parameters
        linear_cmd = self.calculate_pid_command(
            linear_error, dt, self.linear_error_sum, self.linear_error_prev,
            self.kp_linear, self.ki_linear, self.kd_linear
        )
        self.linear_error_sum = linear_cmd[1]
        self.linear_error_prev = linear_error

        angular_cmd = self.calculate_pid_command(
            angular_error, dt, self.angular_error_sum, self.angular_error_prev,
            self.kp_angular, self.ki_angular, self.kd_angular
        )
        self.angular_error_sum = angular_cmd[1]
        self.angular_error_prev = angular_error

        # Create and publish velocity command
        cmd = Twist()
        cmd.linear.x = linear_cmd[0]
        cmd.angular.z = angular_cmd[0]

        # Limit velocities
        cmd.linear.x = max(-1.0, min(1.0, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        self.cmd_vel_pub.publish(cmd)

        # Log adaptive control information
        self.get_logger().info(
            f'Adaptive Control - Linear: {cmd.linear.x:.2f}, '
            f'Angular: {cmd.angular.z:.2f}, '
            f'Kp_L: {self.kp_linear:.3f}, Kp_A: {self.kp_angular:.3f}, '
            f'Avg_Err_L: {avg_linear_error:.3f}, Avg_Err_A: {avg_angular_error:.3f}'
        )

    def adapt_parameters(self, avg_linear_error, avg_angular_error):
        """Adapt controller parameters based on performance"""
        # Adapt linear control parameters
        if avg_linear_error > self.performance_threshold:
            # Increase proportional gain if error is high
            self.kp_linear = min(self.kp_linear + self.adaptation_rate,
                                self.kp_linear * (1 + self.max_adaptation))

            # Adjust integral and derivative gains proportionally
            self.ki_linear = min(self.ki_linear + self.adaptation_rate * 0.1,
                                self.ki_linear * (1 + self.max_adaptation))
            self.kd_linear = min(self.kd_linear + self.adaptation_rate * 0.05,
                                self.kd_linear * (1 + self.max_adaptation))
        else:
            # Decrease gains if error is low (to avoid oscillation)
            self.kp_linear = max(self.kp_linear - self.adaptation_rate * 0.5,
                                self.kp_linear * (1 - self.max_adaptation))
            self.ki_linear = max(self.ki_linear - self.adaptation_rate * 0.05,
                                self.ki_linear * (1 - self.max_adaptation))

        # Adapt angular control parameters
        if avg_angular_error > self.performance_threshold:
            self.kp_angular = min(self.kp_angular + self.adaptation_rate * 2,
                                 self.kp_angular * (1 + self.max_adaptation))
            self.ki_angular = min(self.ki_angular + self.adaptation_rate * 0.2,
                                 self.ki_angular * (1 + self.max_adaptation))
            self.kd_angular = min(self.kd_angular + self.adaptation_rate * 0.1,
                                 self.kd_angular * (1 + self.max_adaptation))
        else:
            self.kp_angular = max(self.kp_angular - self.adaptation_rate,
                                 self.kp_angular * (1 - self.max_adaptation))
            self.ki_angular = max(self.ki_angular - self.adaptation_rate * 0.1,
                                 self.ki_angular * (1 - self.max_adaptation))

    def calculate_linear_error(self):
        """Calculate linear distance error to target"""
        if self.current_pose is None or self.target_pose is None:
            return 0.0

        dx = self.target_pose.position.x - self.current_pose.position.x
        dy = self.target_pose.position.y - self.current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        return distance

    def calculate_angular_error(self):
        """Calculate angular error to target"""
        if self.current_pose is None or self.target_pose is None:
            return 0.0

        # Calculate desired angle to target
        dx = self.target_pose.position.x - self.current_pose.position.x
        dy = self.target_pose.position.y - self.current_pose.position.y
        desired_angle = np.arctan2(dy, dx)

        # Get current orientation (simplified)
        current_angle = 0.0  # Simplified

        # Calculate angle error
        angle_error = desired_angle - current_angle

        # Normalize angle error
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        return angle_error

    def calculate_pid_command(self, error, dt, error_sum, error_prev, kp, ki, kd):
        """Calculate PID command with given parameters"""
        # Proportional term
        p_term = kp * error

        # Integral term
        error_sum += error * dt
        i_term = ki * error_sum

        # Derivative term
        if dt > 0:
            d_term = kd * (error - error_prev) / dt
        else:
            d_term = 0.0

        # Total PID command
        command = p_term + i_term + d_term

        return command, error_sum

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacAdaptiveController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Adaptive controller shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        controller.cmd_vel_pub.publish(cmd)

        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Optimal Control with Trajectory Tracking

Implementing optimal control for precise trajectory following:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.optimize import minimize
import time

class IsaacOptimalController(Node):
    def __init__(self):
        super().__init__('isaac_optimal_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(
            Path,
            '/reference_trajectory',
            self.path_callback,
            10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.pose_callback,
            10
        )

        # Trajectory tracking parameters
        self.lookahead_distance = 0.5  # meters
        self.velocity_profile = []  # Desired velocities along trajectory
        self.path = []
        self.path_index = 0

        # Robot parameters
        self.wheel_base = 0.3  # meters (for differential drive)
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.0

        # Control timer
        self.control_timer = self.create_timer(0.02, self.optimal_control_loop)  # 50Hz

        # State variables
        self.current_pose = None
        self.current_velocity = 0.0

        self.get_logger().info('Isaac optimal controller initialized')

    def path_callback(self, msg):
        """Process reference trajectory"""
        self.path = []
        for pose_stamped in msg.poses:
            self.path.append(np.array([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ]))

        # Calculate velocity profile along trajectory
        self.calculate_velocity_profile()

    def pose_callback(self, msg):
        """Process current pose"""
        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])

    def calculate_velocity_profile(self):
        """Calculate velocity profile for the trajectory"""
        if len(self.path) < 2:
            return

        self.velocity_profile = []

        for i in range(len(self.path)):
            if i == 0:
                # Start with low velocity
                velocity = 0.2
            elif i == len(self.path) - 1:
                # End with low velocity
                velocity = 0.2
            else:
                # Calculate curvature-based velocity
                prev_point = self.path[i-1]
                current_point = self.path[i]
                next_point = self.path[i+1] if i+1 < len(self.path) else current_point

                # Calculate direction vectors
                vec1 = current_point - prev_point
                vec2 = next_point - current_point

                # Calculate angle between segments
                dot_product = np.dot(vec1, vec2)
                norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

                if norms > 0:
                    cos_angle = dot_product / norms
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
                    angle = np.arccos(cos_angle)

                    # Reduce velocity for sharp turns
                    velocity = max(0.2, self.max_linear_vel * (angle / np.pi))
                else:
                    velocity = self.max_linear_vel

            self.velocity_profile.append(min(velocity, self.max_linear_vel))

    def optimal_control_loop(self):
        """Main optimal control loop"""
        if self.current_pose is None or len(self.path) == 0:
            return

        # Find closest point on trajectory
        closest_idx = self.find_closest_point()
        self.path_index = closest_idx

        # Get target point ahead on trajectory
        target_idx = self.find_target_point()

        if target_idx is None:
            return

        # Calculate optimal control command
        cmd = self.calculate_optimal_control(target_idx)

        # Publish command
        twist_cmd = Twist()
        twist_cmd.linear.x = cmd[0]
        twist_cmd.angular.z = cmd[1]

        self.cmd_vel_pub.publish(twist_cmd)

        self.get_logger().info(
            f'Optimal Control - Linear: {twist_cmd.linear.x:.2f}, '
            f'Angular: {twist_cmd.angular.z:.2f}, '
            f'Target Index: {target_idx}'
        )

    def find_closest_point(self):
        """Find the closest point on the trajectory to current position"""
        if len(self.path) == 0:
            return 0

        distances = [np.linalg.norm(self.current_pose - point) for point in self.path]
        closest_idx = np.argmin(distances)

        # Start search from current path index to be more efficient
        if hasattr(self, 'path_index'):
            start_idx = max(0, self.path_index - 5)
            end_idx = min(len(self.path), self.path_index + 5)
            local_distances = [np.linalg.norm(self.current_pose - self.path[i])
                              for i in range(start_idx, end_idx)]
            if local_distances:
                local_closest = np.argmin(local_distances)
                closest_idx = start_idx + local_closest

        return closest_idx

    def find_target_point(self):
        """Find target point at lookahead distance"""
        if self.path_index >= len(self.path):
            return None

        # Start from current closest point and move forward
        current_idx = self.path_index

        # Find point at lookahead distance
        for i in range(current_idx, len(self.path) - 1):
            dist_to_next = np.linalg.norm(self.path[i+1] - self.path[i])
            if dist_to_next > 0:
                # Calculate distance from current pose to next point
                dist_to_next_point = np.linalg.norm(self.path[i+1] - self.current_pose)

                if dist_to_next_point >= self.lookahead_distance:
                    return i+1

        # If no point at exact lookahead distance, return the last point
        return len(self.path) - 1

    def calculate_optimal_control(self, target_idx):
        """Calculate optimal control using optimization"""
        if target_idx >= len(self.path) or self.path_index >= len(self.path):
            return [0.0, 0.0]

        # Get current and target positions
        current_pos = self.current_pose
        target_pos = self.path[target_idx]

        # Calculate desired direction
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance == 0:
            return [0.0, 0.0]

        # Normalize direction
        direction = direction / distance

        # Calculate desired velocity based on trajectory profile
        desired_vel = self.velocity_profile[min(target_idx, len(self.velocity_profile)-1)]

        # Calculate current heading (simplified)
        current_heading = 0.0  # In real implementation, get from pose orientation

        # Calculate angle to target
        target_angle = np.arctan2(direction[1], direction[0])

        # Calculate heading error
        heading_error = target_angle - current_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Use optimization to find best control
        result = self.optimize_control(desired_vel, heading_error)

        return result

    def optimize_control(self, desired_vel, heading_error):
        """Optimize control inputs using scipy"""
        def cost_function(u):
            # Control input: u[0] = linear velocity, u[1] = angular velocity
            linear_vel, angular_vel = u

            # Cost function components
            velocity_cost = (linear_vel - desired_vel)**2
            heading_cost = 10 * heading_error**2  # Weight heading error more
            control_effort_cost = 0.1 * (linear_vel**2 + angular_vel**2)

            # Add penalty for violating constraints
            constraint_penalty = 0
            if abs(linear_vel) > self.max_linear_vel:
                constraint_penalty += 100 * (abs(linear_vel) - self.max_linear_vel)**2
            if abs(angular_vel) > self.max_angular_vel:
                constraint_penalty += 100 * (abs(angular_vel) - self.max_angular_vel)**2

            return velocity_cost + heading_cost + control_effort_cost + constraint_penalty

        # Initial guess
        initial_guess = [desired_vel, heading_error * 2]

        # Bounds for control inputs
        bounds = [(-self.max_linear_vel, self.max_linear_vel),
                  (-self.max_angular_vel, self.max_angular_vel)]

        # Optimize
        result = minimize(cost_function, initial_guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            return result.x
        else:
            # Fallback to simple proportional control
            return [desired_vel, heading_error * 2]

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacOptimalController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Optimal controller shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        controller.cmd_vel_pub.publish(cmd)

        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Control Integration with Perception

Integrating control with perception for intelligent behavior:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import numpy as np

class IsaacPerceptionControl(Node):
    def __init__(self):
        super().__init__('isaac_perception_control')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_detections',
            self.detection_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/navigation_goal',
            self.goal_callback,
            10
        )

        # Control parameters
        self.safety_distance = 0.8  # meters
        self.person_follow_distance = 1.5  # meters
        self.max_speed = 0.5
        self.avoidance_gain = 1.0

        # State variables
        self.current_scan = None
        self.detections = None
        self.navigation_goal = None
        self.following_person = False
        self.follow_target = None

        # Control timer
        self.control_timer = self.create_timer(0.05, self.perception_control_loop)

        self.get_logger().info('Isaac perception control initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.current_scan = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.detections = msg

        # Check if there are people to follow
        if self.should_follow_person():
            self.follow_person()
        else:
            self.following_person = False

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.navigation_goal = msg.pose

    def perception_control_loop(self):
        """Main perception-integrated control loop"""
        cmd = Twist()

        if self.following_person and self.follow_target is not None:
            # Person following mode
            cmd = self.follow_person_control()
        elif self.navigation_goal is not None:
            # Navigation mode
            cmd = self.navigation_control()
        else:
            # Default: stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Apply obstacle avoidance if needed
        if self.current_scan is not None:
            avoidance_cmd = self.obstacle_avoidance()
            if avoidance_cmd is not None:
                # Blend commands or use avoidance if critical
                if self.is_critical_obstacle():
                    cmd = avoidance_cmd
                else:
                    cmd.linear.x = min(cmd.linear.x, avoidance_cmd.linear.x)
                    cmd.angular.z += avoidance_cmd.angular.z * 0.3  # Gentle blending

        # Limit velocities
        cmd.linear.x = max(-self.max_speed, min(self.max_speed, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(
            f'Perception Control - Linear: {cmd.linear.x:.2f}, '
            f'Angular: {cmd.angular.z:.2f}, '
            f'Mode: {"PERSON_FOLLOW" if self.following_person else "NAVIGATION"}'
        )

    def should_follow_person(self):
        """Determine if we should follow a person"""
        if self.detections is None:
            return False

        # Check for person detections with high confidence
        for detection in self.detections.detections:
            for result in detection.results:
                if (result.hypothesis.class_id == 'person' and
                    result.hypothesis.score > 0.7):
                    return True

        return False

    def follow_person(self):
        """Set up person following mode"""
        self.following_person = True
        # In a real implementation, you'd track the specific person
        self.get_logger().info('Switching to person following mode')

    def follow_person_control(self):
        """Generate control commands for person following"""
        cmd = Twist()

        # In a real implementation, you'd use the tracked person's position
        # For this example, we'll simulate following behavior
        cmd.linear.x = 0.3  # Gentle forward motion
        cmd.angular.z = 0.0  # No turning for now

        return cmd

    def navigation_control(self):
        """Generate control for navigation to goal"""
        cmd = Twist()

        # Simple navigation control (in a real system, you'd use path planning)
        cmd.linear.x = 0.4
        cmd.angular.z = 0.0

        return cmd

    def obstacle_avoidance(self):
        """Generate obstacle avoidance commands"""
        if self.current_scan is None:
            return None

        cmd = Twist()
        ranges = np.array(self.current_scan.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            return cmd

        min_range = np.min(valid_ranges)

        if min_range < self.safety_distance:
            # Critical obstacle - emergency avoidance
            cmd.linear.x = 0.0  # Stop

            # Determine turn direction based on obstacle distribution
            num_ranges = len(ranges)
            left_ranges = ranges[num_ranges//2:]
            right_ranges = ranges[:num_ranges//2]

            left_valid = left_ranges[np.isfinite(left_ranges)]
            right_valid = right_ranges[np.isfinite(right_ranges)]

            left_clear = len(left_valid) > 0 and np.min(left_valid) > self.safety_distance
            right_clear = len(right_valid) > 0 and np.min(right_valid) > self.safety_distance

            if left_clear and not right_clear:
                cmd.angular.z = 0.8  # Turn left
            elif right_clear and not left_clear:
                cmd.angular.z = -0.8  # Turn right
            elif left_clear and right_clear:
                # Both sides clear, choose based on which has more clearance
                left_avg = np.mean(left_valid) if len(left_valid) > 0 else 0
                right_avg = np.mean(right_valid) if len(right_valid) > 0 else 0
                cmd.angular.z = 0.8 if left_avg > right_avg else -0.8
            else:
                # Both sides blocked, turn randomly
                cmd.angular.z = 0.8  # Turn left

            return cmd

        elif min_range < self.safety_distance * 2:
            # Moderate obstacle - reduce speed and gentle turning
            cmd.linear.x = max(0.1, self.max_speed * (min_range / (self.safety_distance * 2)))

            # Calculate weighted average for turning
            angles = np.linspace(self.current_scan.angle_min, self.current_scan.angle_max, len(ranges))
            safe_ranges = np.where(ranges > self.safety_distance, ranges, 0)
            if np.sum(safe_ranges) > 0:
                weighted_angle = np.sum(safe_ranges * angles) / np.sum(safe_ranges)
                cmd.angular.z = -weighted_angle * 0.5  # Gentle correction

        return cmd

    def is_critical_obstacle(self):
        """Check if there's a critical obstacle ahead"""
        if self.current_scan is None:
            return False

        # Check forward-facing ranges (simplified)
        ranges = np.array(self.current_scan.ranges)
        forward_start = len(ranges) // 2 - 30
        forward_end = len(ranges) // 2 + 30
        forward_start = max(0, forward_start)
        forward_end = min(len(ranges), forward_end)

        forward_ranges = ranges[forward_start:forward_end]
        valid_forward = forward_ranges[np.isfinite(forward_ranges)]

        if len(valid_forward) > 0:
            return np.min(valid_forward) < self.safety_distance

        return False

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacPerceptionControl()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Perception control shutting down')
    finally:
        # Stop robot
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        controller.cmd_vel_pub.publish(cmd)

        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Isaac ROS Control Systems

1. **Parameter Tuning**: Carefully tune control parameters for your specific robot dynamics
2. **Safety Limits**: Always implement hard safety limits on velocities and accelerations
3. **Performance Monitoring**: Monitor control performance and adjust parameters as needed
4. **Robustness**: Implement fallback behaviors when control objectives cannot be met
5. **Real-time Performance**: Ensure control loops meet real-time requirements

### Physical Grounding and Simulation-to-Real Mapping

When implementing control algorithms:

- **Hardware Acceleration**: Ensure real hardware has compatible NVIDIA GPUs for Isaac ROS optimizations
- **Dynamics Modeling**: Accurately model robot dynamics for effective control
- **Sensor Integration**: Integrate multiple sensors for robust state estimation
- **Environmental Factors**: Account for terrain, friction, and other environmental conditions
- **Safety Systems**: Implement proper safety mechanisms and emergency stops

### Troubleshooting Control Issues

Common control problems and solutions:

- **Oscillation**: Reduce proportional gain or increase derivative gain
- **Slow Response**: Increase proportional gain or adjust integral gain
- **Steady-State Error**: Increase integral gain
- **Instability**: Check sampling rate and reduce gains
- **Poor Tracking**: Verify trajectory feasibility and adjust gains

### Summary

This chapter covered control algorithms using NVIDIA Isaac ROS, focusing on how to implement optimized control systems that leverage Isaac's hardware acceleration and NVIDIA's computing capabilities. You learned about PID control, Model Predictive Control, adaptive control, optimal control, and how to integrate perception with control for intelligent behavior. Isaac ROS control systems provide significant performance benefits for real-time robotics applications, enabling precise and responsive robot motion control. In the next chapter, we'll explore AI processing for robotic systems with Isaac ROS.