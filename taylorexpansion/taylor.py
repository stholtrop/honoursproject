import tensorflow as tf
from .vector_vector_taylor_expansion import taylor_coefficients_vector_vector, pretty_print_taylor_vector
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
        self.batch_expanded_function = batch_vectorize(self.expanded_function)

    def __call__(self, x, batched=None):
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