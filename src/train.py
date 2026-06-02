import os
import json
import logging
import numpy as np
# pyrefly: ignore [missing-import]
from preprocessing import TextPreprocessor
from model import build_lstm_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_pipeline")

# Path Configurations
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
INTENTS_PATH = os.path.join(DATA_DIR, "intents.json")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "chatbot_model.keras")
PREPROCESSOR_SAVE_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

def load_dataset(filepath: str) -> dict:
    """Loads the intents JSON dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Intents file not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_training():
    """Executes the preprocessing and model training pipeline."""
    logger.info("Starting Chatbot Model Training Pipeline...")
    
    # 1. Load data
    dataset = load_dataset(INTENTS_PATH)
    intents = dataset.get("intents", [])
    
    patterns = []
    tags = []
    
    # 2. Extract patterns and tags
    for intent in intents:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(tag)
            
    logger.info(f"Loaded {len(patterns)} patterns across {len(set(tags))} classes/intents.")
    
    # 3. Fit Preprocessor
    preprocessor = TextPreprocessor(max_sequence_len=15)
    preprocessor.fit(patterns, tags)
    
    # 4. Transform features and labels
    X_train = np.array([preprocessor.transform_to_sequence(pat) for pat in patterns])
    y_train = preprocessor.transform_tags_to_one_hot(tags)
    
    logger.info(f"Input features shape: {X_train.shape}")
    logger.info(f"Target labels shape: {y_train.shape}")
    
    # 5. Build Model
    vocab_size = len(preprocessor.vocab)
    num_classes = len(preprocessor.classes)
    
    logger.info("Building LSTM Neural Network Model...")
    model = build_lstm_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_sequence_len=preprocessor.max_sequence_len,
        embedding_dim=64,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.5,
        learning_rate=0.001
    )
    
    model.summary()
    
    # 6. Train Model
    logger.info("Training Model...")
    epochs = 200
    batch_size = 8
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )
    
    logger.info("Training complete.")
    
    # 7. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the preprocessor
    preprocessor.save(PREPROCESSOR_SAVE_PATH)
    
    # Save the Keras model
    model.save(MODEL_SAVE_PATH)
    logger.info(f"Model successfully saved to {MODEL_SAVE_PATH}")
    
    # Simple evaluation on training data
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    logger.info(f"Training Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_training()
