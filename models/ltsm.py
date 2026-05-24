from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model() -> Sequential:

    model = Sequential("LSTM")

    model.add(LSTM(128, return_sequences=True, input_shape=(28, 28))) # adding an LSTM layer with 128 units and input shape of (28, 28) for the sequence of pixel values in the images
    model.add(Dropout(0.3)) # adding a Dropout layer with a dropout rate of 0.3 to prevent overfitting

    model.add(LSTM(64, return_sequences=False)) # adding another LSTM layer with 64 units and return_sequences set to False to output a single vector representation of the input sequence
    model.add(Dropout(0.3))



    model.add(Dense(128, activation="relu")) # adding a Dense layer with 128 neurons
    model.add(BatchNormalization()) 
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu")) # adding a Dense layer with 64 neurons
    model.add(Dropout(0.2))



    model.add(Dense(10, activation="softmax"))
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