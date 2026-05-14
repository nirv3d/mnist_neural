from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model() -> Sequential:
    model = Sequential("DNN")

    model.add(Dense(512, activation="relu", input_shape=(784,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))