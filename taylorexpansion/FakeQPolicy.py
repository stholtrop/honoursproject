import numpy as np
import random

from tf_agents.networks import q_network
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import tf_policy, greedy_policy
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step
from taylorexpansion.approximator import Approximator

class FakeQPolicy(tf_policy.TFPolicy):
    def __init__(self, func, q_policy):
        self.func = func
        self.q_policy = q_policy
        super(FakeQPolicy, self).__init__(
            q_policy.time_step_spec,
            q_policy.action_spec,
            q_policy.policy_state_spec,
            q_policy.info_spec,
            emit_log_probability = q_policy.emit_log_probability,
        )

    def _distribution(self, time_step, policy_state):
        observation = time_step.observation
        q_values = self.func(observation)
        logits = q_values
        
        if self.q_policy._flat_action_spec.minimum != 0:
            distribution = shifted_categorical.ShiftedCategorical(
                logits=logits,
                dtype=self.q_policy._flat_action_spec.dtype,
                shift=self.q_policy._flat_action_spec.minimum)
        else:
            distribution = tfp.distributions.Categorical(
                logits=logits,
                dtype=self.q_policy._flat_action_spec.dtype)

        distribution = tf.nest.pack_sequence_as(self.q_policy._action_spec, [distribution])
        return policy_step.PolicyStep(distribution, policy_state)

def q_net_wrapper(q_net):
    # returns a wrapper that gives back a tensor
    def wrapped(x):
        return q_net(x)[0]
    return wrapped

def get_policy(function, q_policy):
    # returns a wrapper that gives an action given a observation
    return greedy_policy.GreedyPolicy(FakeQPolicy(function, q_policy))

def generate_sample(policy, environment, n_samples):
    points = []
    while len(points) < n_samples:
        time_step = environment.reset()
        points.append(time_step.observation)
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            points.append(time_step.observation)
    return points

        
def get_approximator_policy(q_policy, env, n_samples, n_points, n_terms):
    wrapped_net = q_net_wrapper(q_policy._q_network)
    bounds = np.array(zip(env.observation_spec().minimum, env.observation_spec().maximum))
    points = random.sample(generate_sample(q_policy, env, n_samples), n_points)
    approximator = Approximator(wrapped_net, env.observation_spec().shape[0], env.action_spec().shape[0], points, n_terms, bounds)
    return FakeQPolicy(approximator, q_policy)