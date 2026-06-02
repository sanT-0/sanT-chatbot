import os
import string
import pickle
import logging
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Configure logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Handles text cleaning, tokenization, lemmatization, stop-word removal,
    vocabulary indexing, Bag of Words representation, and text-to-sequence transformation.
    """
    
    def __init__(self, max_sequence_len: int = 15):
        """
        Initializes the preprocessor.
        
        Args:
            max_sequence_len (int): Maximum sequence length for padding/truncating token sequences.
        """
        self.max_sequence_len = max_sequence_len
        self.lemmatizer = None
        self.stop_words = set()
        self.vocab = []
        self.classes = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Download NLTK data and initialize NLP tools
        self._initialize_nlp()

    def _initialize_nlp(self):
        """Downloads required NLTK resources and instantiates lemmatizer."""
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/wordnet.zip', 'wordnet'),
            ('corpora/stopwords.zip', 'stopwords'),
            ('corpora/omw-1.4.zip', 'omw-1.4')
        ]
        
        for check_path, resource_name in resources:
            try:
                nltk.data.find(check_path)
            except LookupError:
                logger.info(f"Downloading NLTK resource: '{resource_name}'")
                nltk.download(resource_name, quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        try:
            raw_stops = set(stopwords.words('english'))
            # Exclude conversational helper words from stop-words list
            # so that intents like "who are you" or "when are you open" are preserved
            exclusions = {
                'who', 'what', 'when', 'where', 'why', 'how', 'you', 'i', 'me',
                'my', 'we', 'us', 'our', 'do', 'does', 'did', 'can', 'could',
                'would', 'should', 'is', 'are', 'am', 'was', 'were', 'be', 'been'
            }
            self.stop_words = raw_stops - exclusions
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords, using fallback. Error: {e}")
            self.stop_words = set()

    def clean_text(self, text: str) -> list[str]:
        """
        Tokenizes, converts to lowercase, removes stop-words and punctuation, and lemmatizes.
        
        Args:
            text (str): Raw input text.
            
        Returns:
            list[str]: Cleaned and lemmatized word tokens.
        """
        if not text or not isinstance(text, str):
            return []
        
        # 1. Lowercase conversion
        text = text.lower()
        
        # 2. Tokenization
        tokens = nltk.word_tokenize(text)
        
        cleaned_tokens = []
        for token in tokens:
            # Remove punctuation
            token_clean = token.translate(str.maketrans('', '', string.punctuation)).strip()
            if not token_clean:
                continue
            
            # 3. Stop-word removal
            if token_clean in self.stop_words:
                continue
                
            # 4. Lemmatization
            lemmatized = self.lemmatizer.lemmatize(token_clean)
            cleaned_tokens.append(lemmatized)
            
        return cleaned_tokens

    def fit(self, patterns: list[str], tags: list[str]):
        """
        Learns the vocabulary from the patterns and registers the unique target tags.
        
        Args:
            patterns (list[str]): List of user query pattern texts.
            tags (list[str]): List of corresponding intent tags.
        """
        vocab_words = set()
        classes_set = set(tags)
        
        for pattern in patterns:
            tokens = self.clean_text(pattern)
            vocab_words.update(tokens)
            
        self.vocab = sorted(list(vocab_words))
        self.classes = sorted(list(classes_set))
        
        # Create vocabulary index map
        # Reserve 0 for Padding, 1 for Out Of Vocabulary (OOV)
        self.word_to_idx = {word: i + 2 for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i + 2: word for i, word in enumerate(self.vocab)}
        
        logger.info(f"Preprocessor fit complete. Vocabulary size: {len(self.vocab)}, Classes count: {len(self.classes)}")

    def transform_to_bow(self, text: str) -> np.ndarray:
        """
        Converts text into a binary Bag of Words vector.
        
        Args:
            text (str): Input sentence.
            
        Returns:
            np.ndarray: 1D binary numpy array of size len(vocab).
        """
        tokens = self.clean_text(text)
        bow = np.zeros(len(self.vocab), dtype=np.float32)
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab.index(token)
                bow[idx] = 1.0
        return bow

    def transform_to_sequence(self, text: str) -> np.ndarray:
        """
        Converts text into an integer token index sequence, padded or truncated
        to self.max_sequence_len.
        
        Args:
            text (str): Input sentence.
            
        Returns:
            np.ndarray: 1D array of integers of length self.max_sequence_len.
        """
        tokens = self.clean_text(text)
        sequence = []
        for token in tokens:
            # 0 is Padding, 1 is OOV
            val = self.word_to_idx.get(token, 1)
            sequence.append(val)
            
        # Truncate if exceeds max length
        if len(sequence) > self.max_sequence_len:
            sequence = sequence[:self.max_sequence_len]
            
        # Pad with 0s at the beginning (pre-padding)
        padded_sequence = np.zeros(self.max_sequence_len, dtype=np.int32)
        if len(sequence) > 0:
            padded_sequence[-len(sequence):] = sequence
            
        return padded_sequence

    def transform_tags_to_one_hot(self, tags: list[str]) -> np.ndarray:
        """
        Converts a list of tags into their one-hot encoded matrix representation.
        
        Args:
            tags (list[str]): List of intent tag labels.
            
        Returns:
            np.ndarray: 2D array of shape (len(tags), len(classes))
        """
        one_hot = np.zeros((len(tags), len(self.classes)), dtype=np.float32)
        for idx, tag in enumerate(tags):
            if tag in self.classes:
                class_idx = self.classes.index(tag)
                one_hot[idx, class_idx] = 1.0
        return one_hot

    def tag_from_index(self, index: int) -> str:
        """Retrieves tag name by index."""
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return "unknown"

    def save(self, filepath: str):
        """Saves the fitted preprocessor instance to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TextPreprocessor':
        """Loads a preprocessor instance from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved preprocessor found at {filepath}")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
