import numpy as np

from tf_agents.networks import q_network
from tf_agents.environments import suite_gym, tf_py_environment

def q_net_wrapper(q_net):
	def wrapped(x):
		return q_net(x)[0]
	return wrapped

def get_policy(function):
	# returns a wrapper that gives an action
	pass

def evaluate_approximation(function):
	env = tf_py_environment.TFPyEnvironment(suite_gym.load("CartPole-v0"))
	time_step = env.reset()
	# make video step 1
	# calculate loss
	
