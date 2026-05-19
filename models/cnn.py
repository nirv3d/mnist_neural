from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model() -> Sequential:

    model = Sequential("CNN")

    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same", input_shape=(28, 28, 1))) # adding a Conv2D layer with 32 filters, a kernel size of 3x3, ReLU activation, and an input shape of (28, 28, 1) for grayscale images
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same")) # adding another Conv2D layer with 64 filters and the same kernel size and activation function
    model.add(BatchNormalization())


    model.add(MaxPooling2D  (pool_size=(2,2))) # adding a MaxPooling2D layer with a pool size of 2x2 to reduce the spatial dimensions of the feature maps
    model.add(Dropout(0.25)) # adding a Dropout layer with a dropout rate of 0.25 to prevent overfitting




    model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same")) # adding a third Conv2D layer with 128 filters
    model.add(BatchNormalization())


    model.add(MaxPooling2D(pool_size=(2,2))) # adding another MaxPooling2D layer to further reduce the spatial dimensions
    model.add(Dropout(0.25))




    model.add(Flatten()) # flattening the 3d feature maps into a 1d vector to feed into the Dense layers






    model.add(Dense(256, activation="relu")) # adding a dense layer with 256 units and ReLU activation
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) # adding a Dropout layer with a dropout rate of 0.5 to prevent overfitting


    model.add(Dense(10, activation="softmax")) # adding an output layer with 10 units and softmax activation for multi-class classification
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat",
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]


