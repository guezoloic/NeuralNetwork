import math
import random

def sigmoid(x: float) -> float:
    return 1/(1 + math.exp(-x))


class Neuron:
    def __init__(self, isize: int) -> None:
        self.isize = isize
        self.weight = [random.uniform(-1, 1) for _ in range(self.isize)]
        self.bias = random.uniform(-1, 1)

    def forward(self, inputs: list[float]) -> float:
        assert len(inputs) == self.isize, "error: incorrect inputs number"
        total = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
        return sigmoid(total)

    def train(self, inputs: list[float], target: float, learning_rate: float = 0.1):
        assert len(inputs) == self.isize, "error: incorrect inputs number"
        
        z = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
        output = sigmoid(z)

        error = output - target
        d_sigmoid = output * (1 - output)
        dz = error * d_sigmoid

        for i in range(self.isize):
            self.weight[i] -= learning_rate * dz * inputs[i]

        self.bias -= learning_rate * dz

class Layer:
    def __init__(self, input_size, output_size):
        self.size = output_size
        self.neurons = [Neuron(output_size) for _ in range(input_size)]
    
    def forward(self, inputs):
        return [n.forward(inputs) for n in self.neurons]

    def train(self, inputs: list[float], targets: list[float], learning_rate: float = 0.1):
        outputs = self.forward(inputs)

        errors = [outputs[i] - targets[i] for i in range(self.size)]

        for i in range(self.neurons):
            self.neurons[i].train(inputs, errors[i], learning_rate)
