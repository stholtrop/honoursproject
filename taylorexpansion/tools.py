import tensorflow as tf
import numpy as np
from collections import Counter

def flatten_function(func, n_input, n_output):
    """
    Simplifies a function with batch input to vector input
    """
    return lambda x: tf.reshape(func(tf.reshape(x, (1, n_input))), (n_output,))

def create_function_vector_vector(coeffs, n_input, n_output, at):
    """
    Creates a function from a Taylor expansion
    """
    def func(x):
        y = np.zeros((n_output,))
        for i in range(n_output):
            total = 0
            for term in coeffs[i]:
                for v, path in term:
                    counts = Counter(path)
                    prod = 1*v
                    for nth, t in counts.items():
                        prod *= (x[nth]-at[nth])**t
                    total += prod
            y[i] = total
        return y
    return func

def batch_vectorize(func, n_output):
    """
    Transforms a function in to accepting a batch of shape (batch_size, ...)
    """
    def fun(x):
        y = np.zeros((len(x), n_output))
        for index, i in enumerate(x):
            y[index] = func(i)
        return y
    return fun