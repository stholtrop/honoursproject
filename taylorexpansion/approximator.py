from .taylor import Taylor
import numpy as np
from tensorflow.keras.losses import MSE

def normalize(un_input, bounds):
    """Normalizes an n-dimensional input to the unit cube

    Args:
        un_input (np.ndarray(n_samples, n_input)): The unnormalized input
        bounds (np.ndarray(n_input, 2)): Bounds for the input normalization
    Returns:
        (np.ndarray(n_samples, n_input)): Normalized input
    """
    return (un_input - bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])

def unnormalize(n_input, bounds):
    return bounds[:, 0] + n_input*(bounds[:, 1]-bounds[:, 0])

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
        self.taylor = []
        for index in range(self.samples.shape[0]):
            self.taylor.append(Taylor(self.network, self.samples[index], self.n_input, self.n_output, self.n_terms, is_batch=True, default_batch=False))
        self.normalized_samples = normalize(self.samples, self.bounds)
    
    def get_closest_point_normalized(self, point_normalized):
        return np.argmin(np.linalg.norm(self.normalized_samples - point_normalized, axis=1))
    
    def get_closest_point_normalized_vectorized(self, points_normalized):
        result = np.empty((points_normalized.shape[0],), dtype=np.int64)
        for i in range(points_normalized.shape[0]):
            result[i] = self.get_closest_point_normalized(points_normalized[i])
        return result

    def __call__(self, input_data):
        """Calls underlying taylor network, in batched mode

        Args:
            input_data (np.ndarray(n_samples, n_input)): The input data on which to call
        Returns:
            np.ndarray(n_samples, n_output)
        """

        evaluation_locations = self.get_closest_point_normalized_vectorized(normalize(input_data, self.bounds))
        result = np.empty((input_data.shape[0], self.n_output))
        for i in range(input_data.shape[0]):
            result[i, :] = self.taylor[evaluation_locations[i]](input_data[i])
        return result
        
    def naive_smoothness(self, input_data):
        """Computes the smoothness of the network in comparison with the Taylor network at selected points

        Args:
            input_data (np.ndarray(n_samples, n_input)): The data to compare the networks on
        Returns:
            float: smoothness_value
        """
        return np.mean(MSE(self.network(input_data), self.__call__(input_data)).numpy())
    
    def naive_smoothness_uniform(self, n_point=10):
        """Computes the smoothness of the network in comparison with the Taylor network at uniformly selected points

        Args:
            n_point (int, optional): Number of points. Defaults to 10.

        Returns:
            float: smoothness value
        """
        input_data = unnormalize(np.random.uniform(size=(n_point, self.n_input)), self.bounds)
        return self.naive_smoothness(input_data)
        


if __name__ == "__main__":
    def test(x):
        # Shape in is 2
        return x**2
    def test2(x):
        return x**3
    X = np.array([[0,0], [1,0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)
    ap = Approximator(test, 2, 2, X, 2, bounds)
    ap2 = Approximator(test2, 2,2, X, 2, bounds)
    test_points = np.array([[0,0], [0.5, 0.7], [-0.1, 0.1], [1,1]], dtype=np.float64)
    print(ap.get_closest_point_normalized_vectorized(normalize(test_points, bounds)))
    print(ap(test_points))
    print(ap.naive_smoothness(test_points))
    print(ap2.naive_smoothness(test_points))
    print(ap.naive_smoothness_uniform(1000))
    print(ap2.naive_smoothness_uniform(1000))