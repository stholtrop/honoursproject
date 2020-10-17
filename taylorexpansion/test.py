import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vector_vector_taylor_expansion import taylor_coefficients_vector_vector, pretty_print_taylor_vector
from tools import flatten_function, create_function_vector_vector, batch_vectorize
from matplotlib import pyplot as plt
import numpy as np

tf.random.set_seed(1)

model = lambda x: 1/(1+x)


new_func = flatten_function(model, 1, 1)

# Feel free to adjust this value and see how the approximation becomes better
n = 6
at = [0.0]
n_input = 1
n_output = 1

coeffs = taylor_coefficients_vector_vector(new_func, n_input, n_output, at, n)
print(pretty_print_taylor_vector(coeffs))
taylor_func = create_function_vector_vector(coeffs, n_input, n_output, at)
batch_taylor_func = batch_vectorize(taylor_func, n_output)
points = np.linspace(-5, 5)

x = np.exp(-points)
batch_points = tf.reshape(x, (-1, 1))
yg = tf.reshape(model(batch_points), (-1,)).numpy()
yw = np.reshape(batch_taylor_func(batch_points), (-1,))

div = points[np.where(np.abs(yw-yg) > 0.01)]
print(div)

plt.plot(points, yw, label="Taylor")
plt.plot(points, yg, label="Model")
plt.ylim((-0.5, 1.5))
plt.grid()
plt.legend()
plt.savefig("test.jpg")