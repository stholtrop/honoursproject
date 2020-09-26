import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vector_vector_taylor_expansion import taylor_coefficients_vector_vector, pretty_print_taylor_vector
from tools import flatten_function, create_function_vector_vector, batch_vectorize
from matplotlib import pyplot as plt
import numpy as np

tf.random.set_seed(1)

model = keras.Sequential([
    layers.Dense(2, name="input", input_shape=(1,), activation="sigmoid"),
    layers.Dense(1, name="output")
    ])


new_func = flatten_function(model, 1, 1)

# Feel free to adjust this value and see how the approximation becomes better
n = 3
coeffs = taylor_coefficients_vector_vector(new_func, 1, 1, [0.0], n)
taylor_func = create_function_vector_vector(coeffs, 1, 1)
batch_taylor_func = batch_vectorize(taylor_func, 1)
points = np.linspace(-1, 1)

batch_points = tf.reshape(points, (-1, 1))
yg = tf.reshape(model(batch_points), (-1,)).numpy()
yw = np.reshape(batch_taylor_func(batch_points), (-1,))

plt.plot(points, yw, label="Taylor")
plt.plot(points, yg, label="Model")
plt.legend()
plt.savefig("test.jpg")