import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

def load_data(model_type: str):

    #loading raw Fashion-MNIST data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    #normalizing piXel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    #one-hot encoding labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test, num_classes=10)

    model_type = model_type.lower()

    if model_type in ("ann", "dnn"):
        X_train = X_train.reshape(-1,784)
        X_test  = X_test.reshape(-1, 784)

    elif model_type == "cnn":
        X_train =X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

    elif model_type in ("rnn", "lstm"):
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        
    elif model_type == "transformer":
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}."
            "Supported types are: 'ann', 'cnn', 'rnn', 'lstm', 'transformer'.")
    
    return (X_train, y_train), (X_test, y_test) #returning preprocessed data ready for model input

def get_input_shape(model_type: str): #In Deep Learning, "Shape" refers to the dimensions and structure of the data array. reorganizing a 28X28 image so that a specific type of AI model can "read" it.
    model_type = model_type.lower()
    shapes = {
        "ann": (784,), #For ANN, we flatten the 28X28 image into a 1D array of 784 piXels.
        "dnn": (784,), #Same as ANN, DNN also eXpects a flat input.
        "cnn": (28, 28, 1), #CNNs eXpect a 3D input: height, width, and channels. Here we have 1 channel for grayscale images.
        "rnn": (28, 28), #RNNs can process sequences, so we treat each row of the image as a time step with 28 features.
        "lstm": (28, 28), #LSTMs are a type of RNN, so they use the same input shape as RNNs.
        "transformer": (28, 28) #Transformers can also process sequences, so we use the same shape as RNNs and LSTMs.
    }
    if model_type not in shapes:
        raise ValueError(
            f"Unsupported model type: {model_type}."
            "Supported types are: 'ann', 'cnn', 'rnn', 'lstm', 'transformer'.")
    return shapes[model_type] #This function returns the eXpected input shape for the specified model type, which is crucial for building the architecture of the neural network correctly.
