import numpy as np

def generate_random_2d_points(num_points=100):
    """
    Generate an array of random 2D points within a [0, 1] range.

    Parameters:
    - num_points: Number of 2D points to generate.

    Returns:
    - An array of shape (num_points, 2) with random values between 0 and 1.
    """
    return np.random.rand(num_points, 2)
