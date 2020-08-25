import tensorflow as tf
import numpy as np
from collections import deque
from math import factorial

def partial_derivative(func, at, path):
    """
    Calculates the partial derivative of a function which takes a vector and outputs a scalar
    func: The function
    at: tuple/tensor/array of point
    path: ordered list of which variables to differentiate to
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
    for i in path:
        last_tape = tapes.pop()
        diffs = last_tape.gradient(diffs, x)[i]
        last_tape.__exit__(None, None, None)
    return diffs

if __name__ == "__main__":
    def simple_func(x):
        # Can be any function or composition of which tensorflow knows the derivatives
        return x[0]**2*x[1]
    derivative = partial_derivative(simple_func, [1.0, 3.0], [0, 0])
    print(derivative)
