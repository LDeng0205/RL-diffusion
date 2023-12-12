import numpy as np

def oval_scaled_distance(x, y, oval_center, oval_axes, oval_rotation):
    """
    Calculate the distance from a point (x, y) to the oval center, scaled by the oval's axes
    and rotated by the oval's angle.

    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param oval_center: (ox, oy) the center of the oval
    :param oval_axes: (oa, ob) the axes of the oval
    :param oval_rotation: rotation angle of the oval in degrees
    :return: modified distance considering the oval shape
    """
    ox, oy = oval_center
    oa, ob = oval_axes
    theta = np.radians(oval_rotation)

    # Rotate the point back
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_rotated = cos_theta * (x - ox) + sin_theta * (y - oy)
    y_rotated = -sin_theta * (x - ox) + cos_theta * (y - oy)

    # Scale the rotated coordinates
    x_scaled = x_rotated / oa
    y_scaled = y_rotated / ob

    # Return the Euclidean distance of the scaled, rotated point
    return np.sqrt(x_scaled**2 + y_scaled**2)

def exponential_oval_decay(x, y, A, k, oval_center, oval_axes, oval_rotation):
    """
    Exponential decay function that considers an oval-shaped decay pattern.

    :param x, y: coordinates of the point
    :param A: maximum value of the function at the center
    :param k: decay rate constant
    :param oval_center: center of the oval
    :param oval_axes: axes of the oval
    :param oval_rotation: rotation of the oval
    :return: value of the function at (x, y)
    """
    distance = oval_scaled_distance(x, y, oval_center, oval_axes, oval_rotation)
    return A * np.exp(-k * distance)

def is_point_in_rotated_oval(x, y, oval_center, oval_axes, oval_rotation):
    ox, oy = oval_center
    a, b = oval_axes
    theta = np.radians(oval_rotation)  # Convert to radians if necessary

    # Rotate the point
    x_rotated = (x - ox) * np.cos(theta) + (y - oy) * np.sin(theta)
    y_rotated = -(x - ox) * np.sin(theta) + (y - oy) * np.cos(theta)

    # Check if the point is inside the oval
    return ((x_rotated**2 / a**2) + (y_rotated**2 / b**2)) <= 1

def get_goodness_slight_overlap(x, y):
    oval_center = (0.4*4, 0.3*4)
    oval_axes = (0.1*4, 0.2*4)
    oval_rotation = 10

    return exponential_oval_decay(x, y, 10, 1, oval_center, oval_axes, oval_rotation)

def get_goodness_no_overlap(x, y):
    oval_center = (0*4, 0*4)
    oval_axes = (0.1*4, 0.2*4)
    oval_rotation = 10

    return exponential_oval_decay(x, y, 10, 1, oval_center, oval_axes, oval_rotation)

def get_goodness_almost_overlap(x, y):
    oval_center = (0.25*4, 0.25*4)
    oval_axes = (0.1*4, 0.2*4)
    oval_rotation = 10

    return exponential_oval_decay(x, y, 10, 1, oval_center, oval_axes, oval_rotation)

def get_goodness_almost_overlap_top(x, y):
    oval_center = (0.3*4, 0.85*4)
    oval_axes = (0.2*4, 0.1*4)
    oval_rotation = 10

    return exponential_oval_decay(x, y, 10, 1, oval_center, oval_axes, oval_rotation)

def get_goodness_almost_overlap_right(x, y):
    oval_center = (0.75*4, 0.6*4)
    oval_axes = (0.1*4, 0.2*4)
    oval_rotation = 10

    return exponential_oval_decay(x, y, 10, 1, oval_center, oval_axes, oval_rotation)

def get_goodness_almost_overlap_step(x, y):
    oval_center = (0.25*4, 0.25*4)
    oval_axes = (0.1*4, 0.2*4)
    oval_rotation = 10

    booleans = is_point_in_rotated_oval(x, y, oval_center, oval_axes, oval_rotation)

    return np.where(booleans, 10, 0)

def get_goodness_move_up(x, y):
  return np.exp(y)

def get_reward(state, end_timestep):
    # compute distance to prefered points and disliked points, return average
    position, timestep = state[:2], state[2]    

    if timestep.item() == end_timestep:
        return get_goodness_almost_overlap_top(position[0].item(), position[1].item())

    return 0
    # return 1 / (position[1] + 1e-6)
