from network import *

def data(size:int, max_val: int):
    def int_to_bits(n: int):
        return [(n >> i) & 1 
            for i in reversed(range(size))
        ]
   
    return [(int_to_bits(i),[i / max_val]) 
        for i in range(max_val + 1)
    ]

def train_network(network: NeuralNetwork, epochs=10000, learning_rate=0.1, 
                  verbose: bool = False, size_data: int = 8, max_val: int = 255):
    
    train_data = data(size_data, max_val)
    
    for epoch in range(epochs):
        for bits, target in train_data:
            network.backward(bits, target, learning_rate)

        if verbose and epoch % 100 == 0:
            output = network.forward(bits)[0]
            loss = (output - target[0]) ** 2

            print(f"Epoch: {epoch}, Loss: {loss:.6f}")

def main():
    size = 8
    max_val = (1 << size) - 1

    network = NeuralNetwork([8, 16, 1])

    print("Start training...")
    train_network(network, verbose=True, size_data=size, epochs=45_000)
    print("End training...")

    while True:
        string = input("Enter 8 bit number (ex: 01101001) or 'quit' to close: ") \
            .strip().lower()
        
        if (string == 'quit'): break
        if (len(string) != 8 or any (char not in '01' for char in string)):
            print("Error: please enter exactly 8 bits (only 0 or 1).") 
            continue
        
        bits_input = [int(char) for char in string]
        output = network.forward(bits_input)[0] * max_val

        print(f"Estimated value: {output} (approx: {round(output)})\n")

if __name__ == "__main__":
    main()
