import os
import json
import random
import logging
# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

from datetime import datetime
import numpy as np
from preprocessing import TextPreprocessor
import tensorflow as tf

# Suppress TensorFlow logging to keep console clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class ChatbotEngine:
    """
    Core engine for loading the trained LSTM chatbot model, running inference on
    user inputs, managing conversation context, and logging chat history.
    """
    
    def __init__(
        self, 
        model_path: str, 
        preprocessor_path: str, 
        intents_path: str,
        log_dir: str,
        confidence_threshold: float = 0.60
    ):
        """
        Initializes the chatbot engine.
        
        Args:
            model_path (str): Path to the trained .keras model file.
            preprocessor_path (str): Path to the saved .pkl preprocessor file.
            intents_path (str): Path to the intents.json file.
            log_dir (str): Directory where conversation logs will be written.
            confidence_threshold (float): Minimum confidence required to accept an intent.
        """
        self.confidence_threshold = confidence_threshold
        self.context = None  # Tracks the active conversation context tag
        
        # Load intents database
        if not os.path.exists(intents_path):
            raise FileNotFoundError(f"Intents dataset not found at {intents_path}")
        with open(intents_path, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)
            
        # Load Preprocessor
        self.preprocessor = TextPreprocessor.load(preprocessor_path)
        
        # Load Keras Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Set up conversation logging
        os.makedirs(log_dir, exist_ok=True)
        self.log_filepath = os.path.join(log_dir, "conversation.log")
        self._setup_conversation_logger()

    def _setup_conversation_logger(self):
        """Sets up a dedicated file logger for conversation history."""
        self.chat_logger = logging.getLogger("chat_logger")
        self.chat_logger.setLevel(logging.WARNING)
        self.chat_logger.propagate = False
        
        # Clear existing handlers to avoid duplicates if re-initialized
        if self.chat_logger.hasHandlers():
            self.chat_logger.handlers.clear()
            
        file_handler = logging.FileHandler(self.log_filepath, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.chat_logger.addHandler(file_handler)

    def _log_conversation(self, user_query: str, predicted_intent: str, confidence: float, response: str):
        """Logs conversation step to the history file."""
        log_entry = (
            f"User: \"{user_query}\" | "
            f"Intent: {predicted_intent} (Conf: {confidence:.2f}) | "
            f"Context Before: {self.context} | "
            f"Response: \"{response}\""
        )
        self.chat_logger.info(log_entry)

    def validate_input(self, text: str) -> tuple[bool, str]:
        """
        Validates the user input.
        
        Returns:
            tuple[bool, str]: (is_valid, error_or_warning_message)
        """
        if not text:
            return False, "Your query is empty. Please type something!"
        
        text_clean = text.strip()
        if len(text_clean) == 0:
            return False, "Your query contains only spaces. Please type a message!"
            
        if len(text_clean) > 500:
            return False, "Your query is too long. Please keep it under 500 characters."
            
        return True, ""

    def get_response(self, user_input: str) -> dict:
        """
        Processes user query, predicts intent, evaluates context, and returns a response.
        
        Args:
            user_input (str): The raw text message from the user.
            
        Returns:
            dict: Contains 'response', 'intent', 'confidence', and 'context' keys.
        """
        # 1. Validate input
        is_valid, err_msg = self.validate_input(user_input)
        if not is_valid:
            return {
                "response": err_msg,
                "intent": "validation_error",
                "confidence": 1.0,
                "context": self.context
            }

        try:
            # 2. Preprocess text into token indices sequence
            seq = self.preprocessor.transform_to_sequence(user_input)
            # Reshape sequence for model input: (1, max_sequence_len)
            model_input = np.expand_dims(seq, axis=0)

            # 3. Model inference
            predictions = self.model.predict(model_input, verbose=0)[0]
            
            # 4. Rank predicted intents
            # Sort indices by probability in descending order
            ranked_indices = np.argsort(predictions)[::-1]
            
            selected_intent = None
            selected_conf = 0.0
            
            # Iterate through intents by predicted confidence to match context filters
            for idx in ranked_indices:
                intent_tag = self.preprocessor.tag_from_index(idx)
                confidence = float(predictions[idx])
                
                # Fetch matching intent configuration from intents json
                intent_conf = next((i for i in self.intents_data["intents"] if i["tag"] == intent_tag), None)
                
                if not intent_conf:
                    continue
                
                # Check for context requirements
                context_filter = intent_conf.get("context_filter")
                
                if context_filter:
                    # If this intent requires a context, trigger it only if it matches active context
                    if self.context == context_filter:
                        selected_intent = intent_conf
                        selected_conf = confidence
                        break
                    else:
                        # Skip this intent since context filter doesn't match
                        continue
                else:
                    # If no context is required, we can select it if we haven't found a context-specific one
                    # But we only select it if the current active context is null or if this is the highest candidate
                    # Let's assign it if we don't find any context-matching intent
                    if selected_intent is None:
                        selected_intent = intent_conf
                        selected_conf = confidence
            
            # 5. Handle confidence thresholding
            if selected_intent is None or selected_conf < self.confidence_threshold:
                response = "I'm sorry, I didn't quite catch that. Could you rephrase your question?"
                predicted_tag = "unknown"
                conf_to_log = selected_conf if selected_intent else 0.0
                
                # We do not modify the context in case of fallback, allowing the user to try again
            else:
                # 6. Retrieve appropriate response
                response = random.choice(selected_intent["responses"])
                predicted_tag = selected_intent["tag"]
                conf_to_log = selected_conf
                
                # Update context state
                context_set = selected_intent.get("context_set")
                if context_set:
                    self.context = context_set
                elif selected_intent.get("context_filter"):
                    # If the selected intent fulfilled a context filter, clear the context
                    self.context = None

            # 7. Log conversation step
            self._log_conversation(user_input, predicted_tag, conf_to_log, response)
            
            return {
                "response": response,
                "intent": predicted_tag,
                "confidence": conf_to_log,
                "context": self.context
            }

        except Exception as e:
            # Safe recovery from runtime errors
            error_response = "I encountered an internal error. Please try again."
            self.chat_logger.error(f"Error handling query '{user_input}': {str(e)}")
            return {
                "response": error_response,
                "intent": "system_error",
                "confidence": 0.0,
                "context": self.context
            }
