import os, sys
# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import ChatbotEngine

def run_test():
    # Paths (adjust if needed)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    MODEL_PATH = os.path.join(MODELS_DIR, 'chatbot_model.keras')
    PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    INTENTS_PATH = os.path.join(DATA_DIR, 'intents.json')

    # Instantiate engine
    engine = ChatbotEngine(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        intents_path=INTENTS_PATH,
        log_dir=LOGS_DIR,
        confidence_threshold=0.6
    )
    # Sample query
    query = "Hello"
    result = engine.get_response(query)
    print('Test query result:', result)

if __name__ == '__main__':
    run_test()
