import tensorflow as tf
import numpy as np
from collections import deque
from math import factorial

def nth_derivative(func, at, n):
    """
    Calculate the nth derivative of a scalar function
    func: The function
    at: At which point
    n: How deep
    """
    # Create a variable which will track the operations
    x = tf.Variable(at, dtype=tf.float32)
    # Create a deque to store the layers of gradient tapes in
    tapes = deque()
    # contract, go deep inside
    for _ in range(n):
        new_tape = tf.GradientTape()
        # Omit the use of the `with`-statement by using its internal methods directly
        new_tape.__enter__()
        tapes.append(new_tape)
    # Inside and calculate
    y = func(x)
    # expand, go back up
    diffs = y
    for _ in range(n):
        last_tape = tapes.pop()
        diffs = last_tape.gradient(diffs, x)
        # Exit the hypothetical `with`-statement
        # Callbacks and aliases do not matter, hence `None`
        last_tape.__exit__(None, None, None)
    return diffs

def taylor_coefficients(func, at, n_terms):
    """
    Finds the Taylor coefficients of a scalar function
    func: The function
    at: Arround which point to Taylor expand
    n_terms: The number of desired coefficients, zeroth term included
    """
    return [nth_derivative(func, at, i).numpy()/factorial(i) for i in range(n_terms)]

def pretty_print(coefficients):
    """
    Print a list of coefficients in a nice way with its powers
    """
    result = "Expansion: "
    for index, i in enumerate(coefficients):
        result += f"{i}*x^{index} "
        if index != len(coefficients)-1:
            result += "+ "
    print(result)

if __name__ == "__main__":
    def simple_func(x):
        # Can be any function or composition of which tensorflow knows the derivatives
        return tf.math.sin(x)
    coefficients = taylor_coefficients(simple_func, 0, 10)
    print()
    pretty_print(coefficients)
