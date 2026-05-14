from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam 

def build_model() -> Sequential: # this code is for building a simple ANN model using Keras Sequential API
    model = Sequential("ANN") # create a Sequential model with the name "ANN"

    model.add(Dense(256, activation="relu", input_shape=(756,))) # add a Dense layer with 256 units, ReLU activation, and input shape of (756,) in input layer
    model.add(Dropout(0.2)) # add a Dropout layer with a dropout rate of 0.2 to prevent overfitting

    model.add(Dense(128, activation="relu")) # adding another Dense layer with 128 units and ReLU activation because it is a common practice to have multiple hidden layers in an ANN model
    model.add(Dropout(0.2)) # add another Dropout layer

    model.add(Dense(10, activation="softmax")) # adding an output layer with 10 units and softmax converts the output to a probability distribution over 10 classes