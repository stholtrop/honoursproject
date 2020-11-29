import taylorexpansion.FakeQPolicy as FQP

from tf_agents.environments import suite_gym, tf_py_environment
import tensorflow as tf

import cartpole.DQNcartpole as dqn

# q_policy = tf.compat.v2.saved_model.load('q_policy')
env_name = 'CartPole-v1'
py_env = suite_gym.load(env_name)
env = tf_py_environment.TFPyEnvironment(py_env)

q_policy = dqn.agent.policy.wrapped_policy

approximator_policy = FQP.get_approximator_policy(q_policy, env, 1000, 10, 1)

FQP.compute_avg_return(env, approximator_policy)
