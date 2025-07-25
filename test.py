import numpy as np

def calculate_weight_compensation(weight, length, R_tip, gravity=np.array([0, 0, -9.81])):
    """
    Calculate the force compensation for the robot's weight based on its orientation.
    
    Args:
        weight (float): Mass of the robot in kg
        length (float): Distance from force sensor to center of mass in meters
        R_tip (3x3 numpy array): Rotation matrix representing the robot's orientation
        gravity (3x1 numpy array): Gravity vector (default is [0, 0, -9.81] m/s²)
    
    Returns:
        compensation_force (3x1 numpy array): Force vector to compensate for weight [Fx, Fy, Fz] in Newtons
        compensation_torque (3x1 numpy array): Torque vector to compensate for weight [Tx, Ty, Tz] in N·m
    """
    # Calculate gravitational force in world frame
    gravity_force = weight * gravity  # [0, 0, -weight*9.81] in world frame
    
    # Transform gravity force to robot's tip frame
    F_tip = R_tip.T @ gravity_force  # R^T * F_world
    
    # Calculate torque due to weight (τ = r × F)
    r = np.array([0, 0, length])  # Assuming COM is along z-axis of tip frame
    τ_tip = np.cross(r, F_tip)
    
    return F_tip, τ_tip

def get_compensated_force(raw_force, weight, length, R_tip):
    """
    Get the compensated force by removing the effect of robot's weight.
    
    Args:
        raw_force (3x1 numpy array): Force reading from sensor [Fx, Fy, Fz]
        weight (float): Mass of the robot in kg
        length (float): Distance from sensor to COM in meters
        R_tip (3x3 numpy array): Rotation matrix of robot's tip
    
    Returns:
        compensated_force (3x1 numpy array): Force with weight compensation [Fx, Fy, Fz]
    """
    F_weight, _ = calculate_weight_compensation(weight, length, R_tip)
    return raw_force - F_weight
    while True:
        force = sen.receive_data()
        print(force)
        time.sleep(0.01)
