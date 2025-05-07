import math
import random

class Neuron:
    def __init__(self, isize: int) -> None:
        self.isize = isize
        self.weight = [random.uniform(0, 1) for _ in range(self.isize)]
        self.bias = random.uniform(0, 1)

    def forward(self, inputs: list) -> float:
        assert len(inputs) == self.isize, "error: incorrect inputs number"
        total = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
        return self.sigmoid(total)
    
    def sigmoid(self, x: float) -> float:
        return 1/(1 + math.exp(-x))

    # target needs to be between 0 and 1
    def train(self, inputs: list, target: float, learning_rate: float = 0.1):
        z = sum(self.weight[i] * inputs[i] for i in range(self.isize)) + self.bias
        output = self.sigmoid(z)

        error = output - target
        d_sigmoid = output * (1 - output)
        dz = error * d_sigmoid

        for i in range(self.isize):
            self.weight[i] -= learning_rate * dz * inputs[i]

        self.bias -= learning_rate * dz
