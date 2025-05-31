import math
import random

# transform all numbers between 0 and 1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# sigmoid's derivation
def sigmoid_deriv(x): 
    y = sigmoid(x)
    return y * (1 - y)

# neuron class
class Neuron:
    """
    z: linear combination of inputs and weights plus bias (pre-activation)
    y : output of the activation function (sigmoid(z))
    w : list of weights, one for each input
    """
    def __init__(self, isize):
        # number of inputs to this neuron
        self.isize = isize
        # importance to each input
        self.weight = [random.uniform(-1, 1) for _ in range(self.isize)]
        # importance of the neuron
        self.bias = random.uniform(-1, 1)

        # last z (linear combination) value
        self.z = 0
        # last output sigmoid(z)
        self.last_output = 0

    def forward(self, x):
        # computes the weighted sum of inputs and add the bias
        self.z = sum(w * xi for w, xi in zip(self.weight, x)) + self.bias
        # normalize the output between 0 and 1
        self.last_output = sigmoid(self.z)
        return self.last_output
    
    # adjust weight and bias
    def backward(self, x, dcost_dy, learning_rate):
        """
        x               : list of input values to the neuron  
        dcost_dy        : derivate of the cost function `(2 * (output - target))`
        learning_rate   : learning factor (adjust the speed of weight/bias change during training)

        weight -= learning_rate * dC/dy * dy/dz * dz/dw
        bias   -= learning_rate * dC/dy * dy/dz * dz/db
        """
        # dy/dz: derivate of the sigmoid activation
        dy_dz = sigmoid_deriv(self.z)
        # dz/dw = x
        dz_dw = x
        # dz/db = 1
        dz_db = 1

        for i in range(self.isize):
            # update all weights by `learning_rate * cost * derivate sigmoid * dz/dw`
            self.weight[i] -= learning_rate * dcost_dy * dy_dz * dz_dw[i]

        # update bias by`learning_rate * cost * derivate sigmoid * dz/db`
        self.bias -= learning_rate * dcost_dy * dy_dz * dz_db

#     def forward(self, inputs: list[float]) -> float:
#         assert len(inputs) == self.isize, "error: incorrect inputs number"
#         total = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
#         return sigmoid(total)

#     def train(self, inputs: list[float], target: float, learning_rate: float = 0.1):
#         assert len(inputs) == self.isize, "error: incorrect inputs number"
        
#         z = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
#         output = sigmoid(z)

#         error = output - target
#         d_sigmoid = output * (1 - output)
#         dz = error * d_sigmoid

#         for i in range(self.isize):
#             self.weight[i] -= learning_rate * dz * inputs[i]

#         self.bias -= learning_rate * dz

# class Layer:
#     def __init__(self, input_size, output_size):
#         self.size = output_size
#         self.neurons = [Neuron(output_size) for _ in range(input_size)]
    
#     def forward(self, inputs):
#         return [n.forward(inputs) for n in self.neurons]

#     def train(self, inputs: list[float], targets: list[float], learning_rate: float = 0.1):
#         outputs = self.forward(inputs)

#         errors = [outputs[i] - targets[i] for i in range(self.size)]

#         for i in range(self.neurons):
#             self.neurons[i].train(inputs, errors[i], learning_rate)
