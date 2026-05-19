from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model() -> Sequential:

    model = Sequential("CNN")

    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1))) # adding a Conv2D layer with 32 filters, a kernel size of 3x3, ReLU activation, and an input shape of (28, 28, 1) for grayscale images
    model.add(BatchNormalization())