import numpy as np
import imageio

from tf_agents.networks import q_network
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import tf_policy, greedy_policy
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step


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

def evaluate_approximation(fake_policy, filename = "taylor_expansion", num_episodes = 5, fps = 30):
    py_env = suite_gym.load("CartPole-v0")
    env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = env.reset()
    # make video step 1
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = env.reset()
            video.append_data(py_env.render())
            while not time_step.is_last():
                action_step = fake_policy.action(time_step)
                time_step = env.step(action_step.action)
                video.append_data(py_env.render())
    # calculate loss

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

if __name__== "__main__":
    from taylorexpansion import Taylor
    import cartpole.DQNcartpole as dqn
    env_name = 'CartPole-v0'
    py_env = suite_gym.load(env_name)
    env = tf_py_environment.TFPyEnvironment(py_env)
    q_policy = dqn.agent.policy.wrapped_policy
    wrapped_net = q_net_wrapper(q_policy._q_network)
    taylor_net = Taylor(wrapped_net, tf.convert_to_tensor([0.0,0.0,0.0,0.0]), 4, 2, 5, True, True)
    fake_policy = FakeQPolicy(taylor_net, q_policy)
    evaluate_approximation(fake_policy)
    print(compute_avg_return(env, fake_policy))