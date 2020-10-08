from taylorexpansion import Taylor
from taylorexpansion.tools import batch_vectorize
import numpy as np
import tensorflow as tf
from tensorflow.math import log, exp
from matplotlib import pyplot as plt
import os

def func(x):
    return log(exp(10*x)+1)/10

batch_func = batch_vectorize(func)
p = np.linspace(-2, 2, num=100)

pr = np.reshape(p, (-1, 1))
yr = tf.reshape(batch_func(pr), (-1,))

num = 60

for index, i in enumerate(np.linspace(-0.6, 0.6, num=num)):
    plt.figure()
    tay = Taylor(batch_func, [i], 1, 1, 7, True, True)

    yt = tay(pr)
    plt.plot(p, yt, label="Taylor")
    plt.plot(p, yr, label="Model")
    plt.legend()
    plt.grid()
    plt.ylim([-1.2, 1.2])
    plt.title(f"Around {i}")
    plt.savefig(f"test{index:d}.jpg")
    print(f"Done with {index}")

os.system(f"bash -c 'convert -delay 20 -loop 0 test{{0..{num-1}}}.jpg anim.gif'")