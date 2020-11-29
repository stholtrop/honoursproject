import tensorflow as tf
from .vector_vector_taylor_expansion import taylor_coefficients_vector_vector, pretty_print_taylor_vector
from .tools import flatten_function, create_function_expression, batch_vectorize

class Taylor:

    def __init__(self, func, at, n_input, n_output, n_terms, is_batch=False, default_batch=False):
        """Initializes a Taylor Network that is callable.

        Args:
            func (lambda x: y): Original network to expand
            at (np.ndarray(n_input)): Point to Taylor expand around
            n_input (int): Number of inputs
            n_output (int): Number of outputs
            n_terms (int): Number of terms in Taylor expansion
            is_batch (bool, optional): If the the function to expand accepts batched input. Defaults to False.
            default_batch (bool, optional): Whether to assume batched input by default. Defaults to False.
        """
        self.at = at
        self.n_input = n_input
        self.n_output = n_output
        self.n_terms = n_terms
        self.default_batch = default_batch
        if is_batch:
            func = flatten_function(func, n_input, n_output)
        self.coeffs = taylor_coefficients_vector_vector(func, n_input, n_output, at, n_terms)
        self.expanded_function = create_function_expression(self.coeffs, n_input, n_output, at)
        self.batch_expanded_function = batch_vectorize(self.expanded_function)

    def __call__(self, x, batched=None):
        """Calculate value at point from Taylor expansion

        Args:
            x (np.ndarray(n_points)): Point to calculate value at
            batched (bool, optional): Defines behaviour, overrides default behaviour. Defaults to None.

        Returns:
            np.ndarray(n_points): Values
        """
        batch = batched if batched else self.default_batch
        if batch:
            return self.batch_expanded_function(x)
        else:
            return self.expanded_function(x)
    
    def __repr__(self):
        result = f"Taylor expansion of a function with {self.n_input} inputs and {self.n_output} outputs"
        result += f"\nExpanded around {self.at} with {self.n_terms} terms"
        result += "\n" + pretty_print_taylor_vector(self.coeffs, self.at)
        return result