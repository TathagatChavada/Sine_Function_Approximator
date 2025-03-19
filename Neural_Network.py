import numpy as np
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, input, output):
        self.weights = np.random.randn(output, input)
        self.biases = np.random.randn(output, 1)

    def forward_propagation(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backward_propagation(self, output_gradient, learning_rate):
        d_error_wrt_weights = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * d_error_wrt_weights
        self.biases -= learning_rate * output_gradient

        return np.dot(self.weights.T, output_gradient)
    

class Tanh_activation_layer():
    def forward_propagation(self, input):
        self.input = input
        return np.tanh(self.input)
    
    def backward_propagation(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, 1 - (np.tanh(self.input) ** 2) )

    

def mean_square_error(actual_output, predicted_output):
    return np.mean(np.power(actual_output - predicted_output, 2))

def d_mean_square_error(actual_output, predicted_output):
    return 2 * (predicted_output - actual_output) / np.size(actual_output)


    
X = np.random.uniform(-10, 10, (100, 1, 1))  
Y = np.sin(X)  


network = [
    Layer(1, 4),
    Tanh_activation_layer(),
    Layer(4, 8),
    Tanh_activation_layer(),
    Layer(8, 11),
    Tanh_activation_layer(),
    Layer(11, 1),
    Tanh_activation_layer()
]

epochs = 50000
learning_rate = 0.0058


plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
sc_actual, = ax.plot(X.flatten(), Y.flatten(), 'go', label='Actual')
sc_pred, = ax.plot([], [], 'ro', label='Predicted')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y = sin(X)')

for epoch in range(epochs):
    error = 0
    predicted_values = []
    
    for x, y in zip(X, Y):
        predicted_output = x
        for layer in network:
            predicted_output = layer.forward_propagation(predicted_output)
        error += mean_square_error(y, predicted_output)

        gradient = d_mean_square_error(y, predicted_output)
        for layer in reversed(network):
            gradient = layer.backward_propagation(gradient, learning_rate)

        predicted_values.append(predicted_output.item())
    
    error /= len(X)
    
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Error: {error}")
        ax.legend([f"Epoch: {epoch} \nError: {error:0.7f}"], loc="upper right", fontsize=10, frameon=True)
        sc_pred.set_xdata(X.flatten())
        sc_pred.set_ydata(predicted_values)
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()