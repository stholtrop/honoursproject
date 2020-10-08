import tensorflow as tf
import numpy as np
from collections import deque
from math import factorial, isnan
from functools import partial, lru_cache
from .scalar_vector_taylor_expansion import sorted_taylor_paths, taylor_paths, pretty

# Make memoized version of factorial
factorial = lru_cache(128)(factorial)

def partial_derivative(func, at, nth_output, path):
    """
    Calculates the partial derivative of a function which takes a vector and outputs a vector
    func: The function
    at: tuple/tensor/array of point
    path: ordered list of which variables to differentiate to
    returns a scalar value
    """
    x = tf.Variable(at)
    tapes = deque()
    # contract
    for _ in range(len(path)):
        new_tape = tf.GradientTape()
        new_tape.__enter__()
        tapes.append(new_tape)
    # Inside and calculate
    y = func(x)[nth_output]
    # expand
    diffs = y
    # An error check is necessary, as too many times differentiating results in an object other than a tensor
    try:
        for i in path:
            last_tape = tapes.pop()
            diffs = last_tape.gradient(diffs, x)[i]
            last_tape.__exit__(None, None, None)
    except TypeError:
        for t in tapes:
            t.__exit__(None, None, None)
        return 0.0
    diffs = diffs.numpy()
    if isnan(diffs):
        diffs = 0.0
    return diffs


def taylor_coefficients_vector_vector(func, n_input, n_output, at, n_terms):
    """
    Calculates the Taylor coefficients of a given function.
    func: function to expand
    n_input: the number of inputs the function takes
    n_output: the number of outputs
    at: point to expand around
    n_terms: number of polynomial terms (highest power - 1)
    Returns a list of lists of lists, each index represents the nth output.
    These sublists contain a normal scalar-vector taylor expansion as lists.
    In turn, these sublists contain tuples, which in turn contain the value of the partial derivative and the path that was taken.
    """
    total = []
    for i in range(n_output):
        coefficients = []
        memo_pd = lru_cache(128)(partial(partial_derivative, func, at, i))
        for power in range(n_terms):
            paths = taylor_paths(power, n_input)
            sorted_paths = sorted_taylor_paths(power, n_input)
            coefficients.append([])
            for p, ps in zip(paths, sorted_paths):
                # Assuming smoothness of the function
                diff = memo_pd(ps)
                coefficients[power].append((diff/factorial(power), p))
        try:
            coefficients[0] = [(func(at)[i].numpy(), ())]
        except AttributeError:
            coefficients[0] = [(func(at)[i], ())]
        total.append(coefficients)
    return total

def pretty_print_taylor_vector(coeffs):
    result = "Total expansion: "
    for index, i in enumerate(coeffs):
        result += f"For output {index+1}\n"
        result += pretty(i) + '\n'
    result = result[:-1]
    return result

if __name__ == "__main__":
    def func(x):
        return [x[0]**2+x[1]*x[0]*2, x[1]]
    coeff = taylor_coefficients_vector_vector(func, 2, 2, [0.0, 0.0], 3)
    print()
    print(coeff)
    print()
    print(pretty_print_taylor_vector(coeff))