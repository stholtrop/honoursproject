# Inspired by https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
import random
import math
import numpy as np
import pyvirtualdisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import gym

import PIL.Image
import imageio
from collections import deque


class DQNAgent:
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0,
            epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01,
            alpha_decay=0.01, batch_size=64, monitor=False):
        # Initialize memory and environment
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v1')

        # Add a monitor
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, './data/cartpole-v1', force=True)
        # Configure paramaters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_log_decay = epsilon_log_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps

        # Create a model
        self.model = Sequential([
            Dense(24, input_shape=(4,), activation='tanh'),
            Dense(48, activation='tanh'),
            Dense(2, activation='linear')
            ])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        print("Summary of the model:")
        self.model.summary()
        print("-"*10)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        # Use epsilon greedy approach
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def calc_epsilon(self, time):
        return max(self.epsilon_min,
                min(self.epsilon, 1.0 - math.log10((time + 1) * self.epsilon_log_decay)))

    def reshape_state(self, state):
        return np.reshape(state, (1,4))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        batch = random.sample(
                self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in batch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])


        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_log_decay


    def run(self):
        scores = deque(maxlen=100)

        for i in range(self.n_episodes):
            print(f"Doing episode: {i}")
            state = self.reshape_state(self.env.reset())
            done = False
            j = 0
            while not done:
                action = self.choose_action(state, self.calc_epsilon(i))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.reshape_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                j += 1
            # Score is number of moves
            scores.append(j)
            print(f"Obtained score: {j}")
            mean_score = np.mean(scores)

            if mean_score >= self.n_win_ticks and i >= 100:
                print(f"Ran {i} episodes. Solved after {i-100} trials")
                return i - 100
            if i % 100 == 0:
                print(f"[Episode {i}] - Mean score over last 100 episodes is {mean_score}")

            self.replay(self.batch_size)

if __name__ == "__main__":
    agent = DQNAgent(n_episodes=200)
    agent.run()
    agent.env.close()


