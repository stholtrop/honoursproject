import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scalar_vector_taylor_expansion import taylor_coefficients_scalar_vector, pretty
from tools import flatten_function

model = keras.Sequential([
    layers.Dense(2, name="input", input_shape=(2,), activation="sigmoid"),
    layers.Dense(1, name="output")
    ])

model.summary()

new_func = flatten_function(model, 2, 1)

coeffs = taylor_coefficients_scalar_vector(new_func, 2, [0.0, 0.0], 2)
print(coeffs)