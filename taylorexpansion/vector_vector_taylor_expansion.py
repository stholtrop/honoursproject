import tensorflow as tf
import numpy as np
from collections import deque
from math import factorial
from functools import partial, lru_cache
from scalar_vector_taylor_expansion import taylor_coefficients_scalar_vector

def taylor_coefficients_vector_vector(func, n_input, n_output, at, n_terms):
    subfuncs = [lambda x: func(x)[i] for i in range(n_output)]
    coefficients = [taylor_coefficients_scalar_vector(sf, n_input, at, n_terms) for sf in subfuncs]
    return coefficients

if __name__ == "__main__":
    def func(x):
        return [x[0]**2, x[1]**2*x[0]**2]
    coeff = taylor_coefficients_vector_vector(func, 2, 2, [1.0, 1.0], 2)
    print(coeff)