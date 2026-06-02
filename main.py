import os
# Suppress TensorFlow and oneDNN verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Show only errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations that emit messages

import sys

# Add src/ to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Silence noisy logs (TensorFlow, warnings, root logger)
import warnings, logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow INFO/WARN
logging.getLogger().setLevel(logging.CRITICAL)

# Attempt to import required modules, handling missing dependencies gracefully
try:
    from chatbot import ChatbotEngine
except ModuleNotFoundError as e:
    if e.name == "nltk":
        print("[WARN] Missing 'nltk' package. Please run 'python install_deps.py' to install dependencies.")
    else:
        print(f"[WARN] Missing module {e.name}. Ensure all dependencies are installed.")
    raise

from train import run_training

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

MODEL_PATH = os.path.join(MODELS_DIR, "chatbot_model.keras")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
INTENTS_PATH = os.path.join(DATA_DIR, "intents.json")
LOG_PATH = os.path.join(LOGS_DIR, "conversation.log")

def print_banner():
    print("=" * 60)
    print("        [BOT]  sanT DEEP LEARNING CHATBOT  [BOT]")
    print("=" * 60)
    print(" Type your message and press Enter.")
    print(" Available Commands:")
    print("   /clear   - Clear the console window")
    print("   /context - Show current context tag")
    print("   /reset   - Reset current context tag")
    print("   /history - Show last 10 log entries")
    print(" quit     - Exit the chat")
    print("=" * 60)

def handle_slash_command(cmd: str, engine: ChatbotEngine) -> bool:
    """Handles chatbot terminal utilities. Returns True if handled."""
    cmd = cmd.lower().strip()
    if cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        print_banner()
        return True
    elif cmd == "/context":
        print(f"👉 Current context: {engine.context}")
        return True
    elif cmd == "/reset":
        engine.context = None
        print("🔄 Context reset successfully.")
        return True
    elif cmd == "/history":
        print("-" * 50)
        print("📜 Recent Chat History:")
        if not os.path.exists(LOG_PATH):
            print(" No logs found yet. Start typing to create logs!")
        else:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.strip())
        print("-" * 50)
        return True
    return False

def main():
    # 1. Auto-train check
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print("[WARN] No trained model detected. Automatic training required.")
        choice = input("Would you like to train the model now? (y/n): ").strip().lower()
        if choice in ("y", "yes", ""):
            print("[TRAIN] Initializing training pipeline...")
            try:
                run_training()
                print("[OK] Model trained and saved successfully!\n")
            except Exception as e:
                print(f"[ERROR] Training failed: {e}")
                sys.exit(1)
        else:
            print("Shutting down. The model must be trained to run the chatbot.")
            sys.exit(0)
            
    # 2. Instantiate Chatbot Engine
    print("[BOT] Loading chatbot brain... Please wait.")
    try:
        engine = ChatbotEngine(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            intents_path=INTENTS_PATH,
            log_dir=LOGS_DIR,
            confidence_threshold=0.60
        )
    except Exception as e:
        print(f"[ERROR] Failed to load chatbot engine: {e}")
        sys.exit(1)
        
    print_banner()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ("quit", "exit"):
                print("sanT Bot: Goodbye! Have a wonderful day!")
                break
                
            if not user_input:
                print("sanT Bot: Please type something so I can help you!")
                continue
                
            # Check for slash commands
            if user_input.startswith("/"):
                if handle_slash_command(user_input, engine):
                    continue
                else:
                    print("sanT Bot: Unknown slash command. Type /clear, /context, /reset, or /history.")
                    continue
            
            # Get response from the engine
            result = engine.get_response(user_input)
            
            # Print response
            print(f"Bot: {result['response']}")
            
            # Print debug metadata unless user requests only reply
            if not any(phrase in user_input.lower() for phrase in ["only replay", "i want only replay", "only response", "i want only response"]):
                intent = result['intent']
                conf = result['confidence']
                ctx = result['context']
                print(f"   [Intent: {intent} | Confidence: {conf:.2f} | Context: {ctx}]")
            print()
            
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    main()
