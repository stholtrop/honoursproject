import tensorflow as tf
from .vector_vector_taylor_expansion import taylor_coefficients_vector_vector
from .tools import flatten_function, create_function_expression, batch_vectorize

class Taylor:

    def __init__(self, func, at, n_input, n_output, n_terms, is_batch=False, default_batch=False):
        self.at =at
        self.n_input = n_input
        self.n_output = n_output
        self.n_terms = n_terms
        self.default_batch = default_batch
        if is_batch:
            func = flatten_function(func, n_input, n_output)
        self.coeffs = taylor_coefficients_vector_vector(func, n_input, n_output, at, n_terms)
        self.expanded_function = create_function_expression(self.coeffs, n_input, n_output, at)
        self.batch_expanded_function = batch_vectorize(self.expanded_function, n_output)

    def __call__(self, x, batched=None):
        batch = batched if batched else self.default_batch
        if batch:
            return self.batch_expanded_function(x)
        else:
            return self.expanded_function(x)