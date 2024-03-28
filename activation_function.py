# Function for nth Fibonacci number 

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
	return np.maximum(alpha*x, x)

def tanh(x):
	return np.tanh(x)


# Random values
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Generate graphs
x = np.linspace(-5, 5, 100)
plt.plot(random_values, relu(random_values), 'sg', label='Sigmoid')

plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.grid(True)
plt.show()

