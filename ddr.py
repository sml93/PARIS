import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class DDR():
  def __init__(self, x_desired, y_desired, dt):
    # Parameters
    self.r = 0.1
    self.L = 0.5
    self.dt = dt

    # PID controller gains
    self.Kp_linear = 1.0
    self.Ki_linear = 0.0
    self.Kd_linear = 0.1
    self.Kp_angular = 6.5
    self.Ki_angular = 0.7
    self.Kd_angular = 0.5
    self.integral_error_linear = 0.0
    self.integral_error_angular = 0.0

    # Initial conditions
    self.x = 0.0
    self.y = 0.0
    self.theta = 0.0
    self.x_desired = x_desired
    self.y_desired = y_desired
    self.v_robot = 0.0
    self.omega_robot = 0.0

  def model(self, v_r_desired, v_l_desired):
    self.v_right = v_r_desired * self.r
    self.v_left = v_l_desired * self.r
    self.v_robot = (self.v_right + self.v_left) / 2
    self.omega_robot = (self.v_right - self.v_left) / self.L

    self.x_dot = self.v_robot * np.cos(self.theta)
    self.y_dot = self.v_robot * np.sin(self.theta)
    self.theta_dot = self.omega_robot

    self.x += self.x_dot * self.dt
    self.y += self.y_dot * self.dt
    self.theta += self.theta_dot * self.dt

    return self.x, self.y, self.theta

  def controller(self):
    x_error = self.x_desired - self.x
    y_error = self.y_desired - self.y
    theta_error = np.arctan2(y_error, x_error) - self.theta

    # Update integral terms
    self.integral_error_linear += x_error * self.dt
    self.integral_error_angular += theta_error * self.dt

    # PID control law
    self.v_control = self.Kp_linear * np.sqrt(x_error**2 + y_error**2) + self.Ki_linear * (self.integral_error_linear) + self.Kd_linear * np.sqrt((x_error**2 + y_error**2))
    self.omega_control = self.Kp_angular * theta_error + self.Ki_angular * (self.integral_error_angular) + self.Kd_angular * theta_error

    # Convert linear and angular velocities to wheel velocities
    self.v_r_desired = (2 * self.v_control + self.L * self.omega_control) / (2 * self.r)
    self.v_l_desired = (2 * self.v_control - self.L * self.omega_control) / (2 * self.r)

    return self.v_r_desired, self.v_l_desired


def main():
  # Simulation parameters
  dt = 0.1  # Time step
  total_time = 30.0
  num_steps = int(total_time / dt)

  # Arrays to store simulation results
  time = np.linspace(0, total_time, num_steps + 1)
  x_vals = np.zeros(num_steps + 1)
  y_vals = np.zeros(num_steps + 1)
  z_vals = np.zeros(num_steps + 1)
  theta_vals = np.zeros(num_steps + 1)

  run = DDR(1.0, 0.5, dt)

  for i in range(num_steps + 1):
    v_r_desired, v_l_desired = run.controller()
    print(v_r_desired, v_l_desired)
    x, y, theta = run.model(v_r_desired, v_l_desired)
    
    x_vals[i] = x
    y_vals[i] = y
    z_vals[i] = 0
    theta_vals[i] = theta

  plt.figure(figsize=(10,6))

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

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot3D(x_vals, y_vals, z_vals)
  plt.grid()
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')
  ax.set_zlim3d(0, 0.2)
  ax.set_title('Trajectory of travel')

  plt.show()


if __name__ == "__main__":
  main()