
import matplotlib.pyplot as plt
import numpy as np

# t = np.linspace(0, 2, 100)
# ee = plt.plot(t, t)
# plt.show()

def compare(old, new):
    assert len(old) == len(new)

    idx = np.linspace(0, len(old), len(old))
    #fig, ax = plt.subplots()
    plt.plot(idx, old-new)
    plt.show()
    # ax.plot(idx, new, label='Old')
    # ax.set_xlabel('x label')  # Add an x-label to the axes.
    # ax.set_ylabel('y label')  # Add a y-label to the axes.
    # ax.set_title("Simple Plot")  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    pass

