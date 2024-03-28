import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

plt.plot(random_values, sigmoid(random_values), label='Sigmoid')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Sigmoid Activation Function')
plt.legend()
plt.grid(True)
plt.show()
