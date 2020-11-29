from .taylor import Taylor
import numpy as np

def normalize(un_input, bounds):
    """Normalizes an n-dimensional input to the unit cube

    Args:
        un_input (np.ndarray(n_samples, n_input)): The unnormalized input
        bounds (np.ndarray(n_input, 2)): Bounds for the input normalization
    Returns:
        (np.ndarray(n_samples, n_input)): Normalized input
    """
    (un_input - bounds[:, 0])/bounds[:, 1]


class Approximator:
    def __init__(self, network, samples):
        """Initialize the Approximator class with a network and samples

        Args:
            network (lambda x(np.ndarray(n_samples, n_input)): y(np.ndarray(n_samples, n_output))): The to be approximated
                        neural network
            samples (np.ndarray(n_points, n_input)): Set of points on which to expand the function
        Returns:
            Approximator object
        """
        pass
    def __call__(self, input_data):
        """Calls underlying taylor network, in batched mode

        Args:
            input_data (np.ndarray(n_samples, n_input)): The input data on which to call
        Returns:
            np.ndarray(n_samples, n_output)
        """
        pass
    def compare(self, input_data):
        """Compares the initially given neural network with the Taylor network

        Args:
            input_data (np.ndarray(n_samples, n_input)): The data to compare the networks on
        Returns:
            float, comparison value
        """
        pass