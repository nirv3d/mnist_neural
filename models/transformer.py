import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Input
)
from tensorflow.keras.optimizers import Adam


def build_model() -> Model:
    """
    Transformer — Self-Attention Model
    ------------------------------------
    Unlike RNN/LSTM which read rows sequentially (one at a time),
    the Transformer looks at ALL 28 rows simultaneously and calculates
    relationships between every pair of rows.

    Key concept — self-attention:
    "How relevant is row 5 to row 18?"
    "How relevant is row 1 to row 28?"
    It answers these questions for every row pair at once.

    This means no vanishing gradient problem — row 1 connects directly
    to row 28 with no chain of recurrent steps in between.

    Architecture:
    - Input projection : maps 28 pixel features → 64 dimensions
    - Multi-Head Attention : 4 heads, each attends to different aspects
    - Feed Forward block : Dense layers applied to each position
    - GlobalAveragePooling : summarises all 28 positions into one vector
    - Dense output : final classification

    Note: uses Functional API instead of Sequential because the
    Transformer block has residual connections (skip connections)
    that Sequential cannot express.

    Input shape : (28, 28) — 28 sequence positions, 28 features each
    Output shape: (10,)    — probability for each clothing class
    """

    # ── input ──────────────────────────────────────────────────────────────
    inputs = Input(shape=(28, 28), name="input")

    # ── input projection ───────────────────────────────────────────────────
    # projects 28 pixel features → 64 dimensions (embedding dimension)
    # gives the model richer representation per row before attention
    x = Dense(64, activation="relu", name="input_projection")(inputs)

    # ── transformer block ──────────────────────────────────────────────────
    # MultiHeadAttention with 4 heads, key dimension 32
    # 4 heads = 4 parallel attention mechanisms, each attends differently
    # one head might focus on top-bottom relationships
    # another might focus on texture consistency across rows
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=32, name="multi_head_attention"
    )(x, x)  # (x, x) means self-attention — query and key are the same

    # dropout on attention output
    attention_output = Dropout(0.1)(attention_output)

    # residual connection + layer norm
    # residual: adds the original x back to attention output
    # prevents information loss and helps gradients flow during training
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # ── feed forward block ─────────────────────────────────────────────────
    # applied independently to each of the 28 positions
    ff_output = Dense(128, activation="relu")(x)
    ff_output = Dropout(0.1)(ff_output)
    ff_output = Dense(64)(ff_output)

    # residual connection + layer norm again
    x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    # ── pooling ────────────────────────────────────────────────────────────
    # averages across all 28 positions → single vector of 64 values
    # summarises the full sequence for classification
    x = GlobalAveragePooling1D()(x)

    # ── classification head ────────────────────────────────────────────────
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation="softmax", name="output")(x)

    # ── build model using functional api ───────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="Transformer")


    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
    