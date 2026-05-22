from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model() -> Sequential:
    model = Sequential("RNN")

    model.add(SimpleRNN(128, activation="tanh", return_sequences=True, input_shape=(28, 28)))

    model.add(SimpleRNN(64, activation="tanh", return_sequences=False))

    model.add(Dropout(0.3))




