import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return np.ones(x.shape) - tanh(x) ** 2


data = np.loadtxt('./dane4.txt')

X = data[:, 0]
X = X.reshape((1, len(X)))

y = data[:, 1]
y = y.reshape((1, len(y)))

# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


W1 = np.random.random_sample((20,1))
#print(W1)
B1 = (5 + 0.5) * np.random.random_sample((20,1)) - 0.5
#print(B1)
W2 = np.random.random_sample((1,20))
#print(W2)
B2 = (5 + 0.5) * np.random.random_sample((1,1)) - 0.5
#print(B2)

epochs = 30000
lr = 0.001
final_output = None
for _ in range(epochs):
    #print(f"epoch with index {_}")
    N1 = tanh(W1 @ X + B1)
    N2 = W2 @ N1 + B2

    E2 = y - N2
    E1 = W2.T @ E2

    dW2 = lr * E2 @ N1.T
    dB2 = lr * E2 @ np.ones(E2.T.shape)
    dW1 = lr * tanh_derivative(N1) * E1 @ X.T
    dB1 = lr * tanh_derivative(N1) * E1 @ np.ones(X.T.shape)

    W2 = W2 + dW2
    B2 = B2 + dB2

    W1 = W1 + dW1
    B1 = B1 + dB1
    final_output = N2

#%%
#print(X, y)
plt.scatter(X[0], y[0])
plt.title('data')
plt.show()
plt.plot(X[0], final_output[0])
plt.title('prediciton curve')
plt.show()

