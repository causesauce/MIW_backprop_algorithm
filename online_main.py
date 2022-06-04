import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return np.ones(x.shape) - x ** 2


data = np.loadtxt('./dane4.txt')

W1 = np.random.random_sample((20, 1))

B1 = (5 + 0.5) * np.random.random_sample((20, 1)) - 0.5

W2 = np.random.random_sample((1, 20))

B2 = (5 + 0.5) * np.random.random_sample((1, 1)) - 0.5

epochs = 2000
lr = 0.01
final_output = None
output = []
for _ in range(epochs):
    for X, y in data:
        # print(X, y)
        N1 = tanh(X * W1 + B1)
        N2 = W2 @ N1 + B2

        E2 = y - N2
        E1 = E2 * W2

        dW2 = lr * E2 * N1
        dB2 = lr * E2
        dW1 = lr * tanh_derivative(N1) * E1 * X
        dB1 = lr * tanh_derivative(N1) * E1

        if _ == (epochs - 1):
            output.append(N2[0,0])

# print(X, y)
plt.scatter(data[:, 0], data[:, 1])
plt.show()
print(output)
plt.plot(data[:, 0], output)
plt.show()
