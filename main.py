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
    z                   : linear combination of inputs and weights plus bias (pre-activation)
    y                   : output of the activation function (sigmoid(z))
    w                   : list of weights, one for each input
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
        """
        x               : list of input values to the neuron
        """
        # computes the weighted sum of inputs and add the bias
        self.z = sum(w * xi for w, xi in zip(self.weight, x)) + self.bias
        # normalize the output between 0 and 1
        self.last_output = sigmoid(self.z)
        return self.last_output
    
    # adjust weight and bias of neuron
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
            # update each weight `weight -= learning_rate * dC/dy * dy/dz * x_i`
            self.weight[i] -= learning_rate * dcost_dy * dy_dz * dz_dw[i]

        # update bias: bias -= learning_rate * dC/dy * dy/dz * dz/db
        self.bias -= learning_rate * dcost_dy * dy_dz * dz_db

        # return gradient vector len(input) dimension
        return [dcost_dy * dy_dz * w for w in self.weight]


class Layer:
    def __init__(self, input_size, output_size):
        """
        input_size      : size of each neuron input
        output_size     : size of neurons
        """
        self.size = output_size
        # list of neurons
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        self.inputs = inputs
        #  compute and return the outputs of all neurons in the layer
        return [neuron.forward(inputs) for neuron in self.neurons]

    # adjust weight and bias of the layer (all neurons)
    def backward(self, dcost_dy_list, learning_rate=0.1):
        # init layer gradient vector len(input) dimention
        input_gradients = [0.0] * len(self.inputs)

        for i, neuron in enumerate(self.neurons):
            dcost_dy = dcost_dy_list[i]
            grad_to_input = neuron.backward(self.inputs, dcost_dy, learning_rate)

            # compute all neuron's gradient inside layer gradient
            # accumulate the input gradients from all neurons
            for j in range(len(grad_to_input)):
                input_gradients[j] += grad_to_input[j]

        # return layer gradient
        return input_gradients