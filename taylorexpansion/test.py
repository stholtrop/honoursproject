import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import numpy as np
from taylorexpansion import Taylor

tf.random.set_seed(1)

# Feel free to adjust this value and see how the approximation becomes better
n = 5
at = [0.0]
n_input = 1
n_output = 1

model = keras.Sequential([
    layers.Dense(5, name="input", input_shape=(n_input,), activation="sigmoid"),
    layers.Dense(5, activation="tanh"),
    layers.Dense(n_output, name="output")
    ])

func = Taylor(model, at, n_input, n_output, n, is_batch=True)

points = np.linspace(-2, 2)

batch_points = tf.reshape(points, (-1, 1))
yg = tf.reshape(model(batch_points), (-1,)).numpy()
yw = np.reshape(func(batch_points, batched=True), (-1,))

plt.plot(points, yw, label="Taylor")
plt.plot(points, yg, label="Model")
plt.legend()
plt.savefig("test.jpg")