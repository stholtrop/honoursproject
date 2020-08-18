import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym



model = keras.Sequential([
    layers.Dense(4, name="input", input_shape=(6,), activation="Sigmoid"),
    layers.Dense(1, name="output")
    ])


"""
x = tf.ones((1,4))
print(x)
y = model(x)
print(y)
"""
model.summary()


env = gym.make("CartPole-v1")


