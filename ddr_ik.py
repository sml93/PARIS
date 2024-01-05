import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.1  # Wheel radius
L = 0.5  # Distance between wheels

# PD Controller gains
Kp_linear = 1.0
Ki_linear = 0.0
Kd_linear = 0.1
Kp_angular = 6.5
Ki_angular = 0.7
Kd_angular = 0.5

# Initial conditions
x = 0.0
y = 0.0
theta = 0.0
x_desired = 1.0  # Desired X position
y_desired = 0.5  # Desired Y position

# Simulation parameters
dt = 0.1  # Time step
total_time = 30.0
num_steps = int(total_time / dt)

# Arrays to store simulation results
time = np.linspace(0, total_time, num_steps + 1)
x_vals = np.zeros(num_steps + 1)
y_vals = np.zeros(num_steps + 1)
theta_vals = np.zeros(num_steps + 1)

# PD Controller function
def pd_controller(x, y, theta, x_desired, y_desired, integral_error_linear, integral_error_angular):
    # Calculate errors
    x_error = x_desired - x
    y_error = y_desired - y
    theta_error = np.arctan2(y_error, x_error) - theta

    # Update integral terms
    integral_error_linear += x_error * dt
    integral_error_angular += theta_error * dt

    # PD control law
    v_control = Kp_linear * np.sqrt(x_error**2 + y_error**2) + Ki_linear * (integral_error_linear) + Kd_linear * np.sqrt((x_error**2 + y_error**2))
    omega_control = Kp_angular * theta_error + Ki_angular * (integral_error_angular) + Kd_angular * theta_error

    # Convert linear and angular velocities to wheel velocities
    v_r_desired = (2 * v_control + L * omega_control) / (2 * r)
    v_l_desired = (2 * v_control - L * omega_control) / (2 * r)

    return v_r_desired, v_l_desired, integral_error_linear, integral_error_angular


# Initialization
integral_error_linear = 0.0
integral_error_angular = 0.0


# Simulation loop
for i in range(num_steps + 1):
    v_r_desired, v_l_desired, integral_error_linear, integral_error_angular = pd_controller(x, y, theta, x_desired, y_desired, integral_error_linear, integral_error_angular)

    # Update robot's state using kinematic equations
    v_right = r * v_r_desired
    v_left = r * v_l_desired

    v_robot = (v_right + v_left) / 2
    omega_robot = (v_right - v_left) / L

    x_dot = v_robot * np.cos(theta)
    y_dot = v_robot * np.sin(theta)
    theta_dot = omega_robot

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    # Store results
    x_vals[i] = x
    y_vals[i] = y
    theta_vals[i] = theta

# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time, np.degrees(theta_vals), label='Orientation')
plt.title('Differential Drive Robot Control')
plt.ylabel('Orientation (degrees)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, x_vals, label='X Position')
plt.ylabel('X Position')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, y_vals, label='Y Position')
plt.xlabel('Time (seconds)')
plt.ylabel('Y Position')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
