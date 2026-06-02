import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(
    vocab_size: int, 
    num_classes: int, 
    max_sequence_len: int, 
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Builds and compiles an LSTM-based sequential model for intent classification.
    
    Args:
        vocab_size (int): Size of vocabulary. Note that the embedding layer will use 
                           vocab_size + 2 to account for padding (0) and OOV (1).
        num_classes (int): Number of intent categories.
        max_sequence_len (int): Input sequence length (padded/truncated query tokens).
        embedding_dim (int): Dimensionality of the dense embedding space.
        lstm_units (int): Number of hidden units in the LSTM layer.
        dense_units (int): Number of units in the intermediate Dense layer.
        dropout_rate (float): Dropout probability for regularization.
        learning_rate (float): Learning rate for the Adam optimizer.
        
    Returns:
        Sequential: Compiled Keras Sequential model.
    """
    model = Sequential([
        # 1. Embedding Layer
        # input_dim: vocab_size + 2 (Padding = 0, OOV = 1)
        # mask_zero=True: Tells the downstream layers to ignore padding tokens (0)
        Embedding(
            input_dim=vocab_size + 2, 
            output_dim=embedding_dim, 
            input_length=max_sequence_len,
            mask_zero=True,
            name="embedding"
        ),
        
        # 2. Spatial Dropout
        # Regularization designed for 1D feature maps (like sequences of word embeddings)
        SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        
        # 3. Bidirectional LSTM
        # Learns temporal patterns from left-to-right and right-to-left
        Bidirectional(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2), name="bidirectional_lstm"),
        
        # 4. Dense Layer with ReLU activation
        Dense(dense_units, activation="relu", name="dense_hidden"),
        
        # 5. Dropout Layer
        Dropout(dropout_rate, name="dropout_regularization"),
        
        # 6. Output Layer with Softmax activation
        Dense(num_classes, activation="softmax", name="output_classification")
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model
