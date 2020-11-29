import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import numpy as np
from taylorexpansion import Taylor

tf.random.set_seed(1)

# Feel free to adjust this value and see how the approximation becomes better
n = 5
at = [0.0, 1.0]
n_input = 2
n_output = 1

model = keras.Sequential([
    layers.Dense(8, name="input", input_shape=(n_input,), activation="sigmoid"),
    layers.Dense(12, activation="sigmoid"),
    layers.Dense(n_output)
    ])

func = Taylor(model, at, n_input, n_output, n, is_batch=True)

points = np.array([np.linspace(-2, 2), np.zeros(50)]).T

batch_points = tf.reshape(points, (-1, 2))
yg = tf.reshape(model(batch_points), (-1,)).numpy()
yw = np.reshape(func(batch_points, batched=True), (-1,))

plt.plot(points[:, 0], yw, label="Taylor")
plt.plot(points[:, 0], yg, label="Model")
plt.legend()
plt.savefig("test.jpg")