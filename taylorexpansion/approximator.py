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
    return (un_input - bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])


class Approximator:
    def __init__(self, network, n_input, n_output, samples, n_terms, bounds):
        """Initialize the Approximator class with a network and samples

        Args:
            network (lambda x(np.ndarray(n_samples, n_input)): y(np.ndarray(n_samples, n_output))): The to be approximated
                        neural network
            n_input (int): number of inputs to the network
            n_output (int): number of outputs of the network
            samples (np.ndarray(n_points, n_input)): Set of points on which to expand the function
            n_terms (int): number of taylor terms for the Approximator
            bounds (np.ndarray(n_input, 2)): Bounds for the input normalization
        Returns:
            Approximator object
        """
        self.network = network
        self.n_input = n_input
        self.n_output = n_output
        self.samples = samples
        self.n_terms = n_terms
        self.bounds = bounds
        # Construct Taylor network
        self.taylor = {}
        for point in self.samples:
            self.taylor[point] = Taylor(self.network, point, self.n_input, self.n_output, self.n_terms, is_batch=True, default_batch=True)

    def __call__(self, input_data):
        """Calls underlying taylor network, in batched mode

        Args:
            input_data (np.ndarray(n_samples, n_input)): The input data on which to call
        Returns:
            np.ndarray(n_samples, n_output)
        """
        
    def compare(self, input_data):
        """Compares the initially given neural network with the Taylor network

        Args:
            input_data (np.ndarray(n_samples, n_input)): The data to compare the networks on
        Returns:
            float, comparison value
        """
        pass