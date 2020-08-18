try:
    import tensorflow
    import gym
    import tqdm
    import matplotlib.pyplot as plt
    import numpy
except ImportError as e:
    print("Failed to import:", e)
else:
    print("Successfully imported all libraries")
