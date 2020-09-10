import tensorflow as tf
import numpy as np
from collections import deque
from math import factorial, isnan
from functools import partial, lru_cache


# Make memoized version of factorial
factorial = lru_cache(128)(factorial)

def partial_derivative(func, at, path):
    """
    Calculates the partial derivative of a function which takes a vector and outputs a scalar
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
    y = func(x)
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


@lru_cache(128)
def taylor_paths(power, n_input):
    """
    Calculate the paths that partial derivatives have to take. Caching is added to minimize time spent in recursive calls
    power: the power required
    n_input: the number of inputs the expanded function requires
    returns a list of tuples, where each tuple indicates the order of differentitation, with 0 the first variable.
    """
    if power == 0:
        return []
    elif power == 1:
        return [(i,) for i in range(n_input)]
    else:
        prev = taylor_paths(power-1, n_input)
        new_paths = []
        for i in prev:
            for j in range(n_input):
                new_paths.append(i + (j,))
        return new_paths


@lru_cache(128)
def sorted_taylor_paths(power, n_input):
    """
    Calculate the Taylot paths, but sort them, as for smooth functions (mathematical sense) 
    order does not matter. 
    power: the power required
    n_input: the number of inputs the expanded function requires
    returns a list of sorted tuples, where each tuple indicates the amount of differentiating to one variable
    """
    paths = taylor_paths(power, n_input)
    for i in range(len(paths)):
        paths[i] = tuple(sorted(list(paths[i])))
    return paths


def taylor_coefficients_scalar_vector(func, n_input, at, n_terms):
    """
    Calculates the Taylor coefficients of a given function.
    func: function to expand
    n_input: the number of inputs the function takes
    at: point to expand around
    n_terms: number of polynomial terms (highest power - 1)
    returns a list of lists, each index represents the power of that element. 
    The sublists contain tuples, which in turn contain the value of the partial derivative and the path that was taken.
    """
    coefficients = []
    memo_pd = lru_cache(128)(partial(partial_derivative, func, at))
    for power in range(n_terms):
        paths = taylor_paths(power, n_input)
        sorted_paths = sorted_taylor_paths(power, n_input)
        coefficients.append([])
        for p, ps in zip(paths, sorted_paths):
            # Assuming smoothness of the function
            diff = memo_pd(ps)
            coefficients[power].append((diff/factorial(power), p))
    coefficients[0] = [(func(at), ())]
    return coefficients


def pretty(coefficients):
    """
    Print a list of coefficients in a nice way with its powers
    """
    from string import ascii_lowercase
    from collections import Counter
    variables = 'xyz' + ascii_lowercase[:-3]
    result = "Expansion: "
    result += str(coefficients[0][0][0])
    if len(coefficients) == 1:
        print(result)
    for i in coefficients[1:]:
        result += " + ("
        for v, p in i:
            result += str(v)
            counts = Counter(p)
            for t, n in counts.items():
                result += "*" + variables[t] + "^" + str(n)
            result += " + "
        result = result[:-3]
        result += ")"
    return result


if __name__ == "__main__":
    def simple_func(x):
        # Can be any function or composition of which tensorflow knows the derivatives
        return tf.math.sin(x[0])
    coeff = taylor_coefficients_scalar_vector(simple_func, 1, [0.0, 0.0], 4)
    print(coeff)
    print(pretty(coeff))