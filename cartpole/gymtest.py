import pyvirtualdisplay
import PIL.Image
from tf_agents.environments import suite_gym
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

env = suite_gym.load("CartPole-v1")
env.reset()
x = PIL.Image.fromarray(env.render())
x.save("test.jpg")
env.close()
