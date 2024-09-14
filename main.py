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

def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return np.where(x == 0, 0.5, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_prime(x):
    pass

class Layer:
    def __init__(self, weight_shape, activation, derivative):
        self.values = np.ones((weight_shape[0], 1))
        self.bias = np.zeros((weight_shape[0], 1))
        self.weights = np.random.rand(weight_shape[0], weight_shape[1])
        self.func = activation
        
    def forward(self, prev_layer):
        self.values = self.func(self.weights @ prev_layer + self.bias)

    def backward(self):
        pass

    def get_weights(self):
        return self.weights

    def get_values(self):
        return self.values
    
    def get_neuron_cnt(self):
        return self.values.shape[0]

def main():
    training_data, test_data = load_data("./Training_Data")

    h_layers = []
    h_layers.append(Layer((IMG_RES_SQ//2, IMG_RES_SQ), ReLU, ReLU_prime))
    for i in range(NO_HIDDEN-1):
        prev_neurons = h_layers[i].get_neuron_cnt()
        cur_neurons = prev_neurons//2
        h_layers.append(Layer((cur_neurons, prev_neurons), ReLU, ReLU_prime))
    
    o_layer = Layer((2, h_layers[-1].get_neuron_cnt()), softmax, softmax_prime)

    # print(training_data[0])
    
    for epoch in range(NO_EPOCH):
        for data in training_data[0]:
            for layer in range(len(h_layers)):
                if layer == 0:
                    h_layers[layer].forward(data)
                else:
                    h_layers[layer].forward(h_layers[layer-1].get_values())

            o_layer.forward(h_layers[-1].get_values())
            if (o_layer.get_values() == HOTDOG).all():
                print("hotdog")
            else:
                print("not hotdog")

if __name__ == "__main__":
    main()