import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

def load_data(model_type: str):

    #loading raw Fashion-MNIST data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #normalizing pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    #one-hot encoding labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test, num_classes=10)

    model_type = model_type.lower()

    if model_type in ("ann", "dnn"):
        x_train = x_train.reshape(-1,784)
        x_test  = x_test.reshape(-1, 784)

    elif model_type == "cnn":
        x_train =x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

    elif model_type in ("rnn", "lstm"):
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)
        
    elif model_type == "transformer":
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)

    else:
