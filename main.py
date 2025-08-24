import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# binary representation of hotdogs and not hotdogs
HOTDOG = np.array([[1], [0]])
NOTDOG = np.array([[0], [1]])

IMG_RES = 64    # resolution images are scaled to (64x64)
IMG_RES_SQ = IMG_RES * IMG_RES
NO_HIDDEN = 10   # number of hidden layers
NO_EPOCH = 5   # number of epochs
LEARNING_RATE = 0.01

def load_data(folder):
    images = []
    one_hot_encodings = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_RES, IMG_RES))
            img = np.reshape(img, (IMG_RES_SQ, 1))
            img = np.float64(img) / 255
            classification = filename.split("_")[0]
            one_hot_enc = np.copy(HOTDOG) if classification == "hotdog" else np.copy(NOTDOG)
            images.append(img)
            one_hot_encodings.append(one_hot_enc)

    training = (images[:int(len(images)*0.8)], one_hot_encodings[:int(len(images)*0.8)])
    test = (images[int(len(images)*0.8):], one_hot_encodings[int(len(images)*0.8):])

    return training, test

def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def ReLU_prime(x: np.ndarray) -> np.ndarray:
    return x > 0

# sigmoid is used for binary classification
# https://www.geeksforgeeks.org/deep-learning/softmax-vs-sigmoid-activation-function/
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(-x)
    return e_x / (1 + e_x)**2

class Layer:
    def __init__(self, weight_shape: tuple, activation: callable, derivative: callable):
        self.z = np.zeros((weight_shape[0], 1)) # helper matrix for backprop
        self.values = np.ones((weight_shape[0], 1))
        self.bias = np.zeros((weight_shape[0], 1))
        self.weights = np.random.rand(weight_shape[0], weight_shape[1]) / 1000

        self.activate = activation
        self.activate_prime = derivative

    def forward(self, prev_layer: np.ndarray):
        self.z = self.weights @ prev_layer + self.bias
        self.values = self.activate(self.z)

    # Good explanation: https://youtu.be/tIeHLnjs5U8?t=313
    def backward(self, prev_layer: np.ndarray, target: np.ndarray = None, next_layer_error: np.ndarray = None, next_weights: np.ndarray = None):
        # Calculate the gradient of the loss with respect to the weights and biases (dCdW and dCdB)
        dZdW = prev_layer
        dZdB = 1
        dAdZ = self.activate_prime(self.z)

        # Calculate errors
        if next_weights is None:
            # Output layer
            dCdA = 2 * (self.values - target)
            error = dCdA * dAdZ
        else:
            # Hidden layer
            error = (next_weights.T @ next_layer_error) * dAdZ

        # Assign new weights and biases
        self.weights -= LEARNING_RATE * (error @ dZdW.T)
        self.bias -= LEARNING_RATE * (error * dZdB)

        return error

    def get_weights(self) -> np.ndarray:
        return np.copy(self.weights)

    def get_values(self) -> np.ndarray:
        return self.values

    def get_neuron_cnt(self) -> int:
        return self.values.shape[0]

def main():
    training_data, test_data = load_data("./Training_Data")

    total_error = 0
    correct = 0
    n_training = len(training_data[0])

    h_layers = []
    h_layers.append(Layer((IMG_RES_SQ//2, IMG_RES_SQ), ReLU, ReLU_prime))
    for i in range(NO_HIDDEN-1):
        prev_neurons = h_layers[i].get_neuron_cnt()
        cur_neurons = prev_neurons//2
        h_layers.append(Layer((cur_neurons, prev_neurons), ReLU, ReLU_prime))
    
    o_layer = Layer((2, h_layers[-1].get_neuron_cnt()), sigmoid, sigmoid_prime)
    
    for epoch in range(NO_EPOCH):
        for data, target in zip(training_data[0], training_data[1]):

            h_layers[0].forward(data)
            for layer in range(1, len(h_layers)):
                h_layers[layer].forward(h_layers[layer-1].get_values())
            o_layer.forward(h_layers[-1].get_values())

            total_error += np.mean(np.square(o_layer.get_values() - target))
            correct += np.argmax(o_layer.get_values()) == np.argmax(target)

            next_weights = o_layer.get_weights()
            error = o_layer.backward(h_layers[-1].get_values(), target=target)
            for layer in range(len(h_layers)-1, 0, -1):
                old_weights = h_layers[layer].get_weights()
                error = h_layers[layer].backward(h_layers[layer-1].get_values(), next_layer_error=error, next_weights=next_weights)
                next_weights = old_weights
            h_layers[0].backward(data, next_layer_error=error, next_weights=next_weights)

        print("Epoch:", epoch)
        print("Error:", total_error/n_training)
        print(f"{correct} correct out of {n_training} \n")
        total_error = 0
        correct = 0

if __name__ == "__main__":
    main()