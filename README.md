# sanT Deep Learning Chatbot

A conversational AI chatbot built with Python, TensorFlow/Keras, and Flask. It supports intent classification, context handling, and a set of predefined intents.

## Features
- Intent classification using a trained LSTM model.
- Context‑aware conversation flows (e.g., booking slots).
- Rich set of FAQs (technology, payments, account creation, etc.).
- Easy to extend with new intents.
- Command‑line interface with slash commands (`/clear`, `/context`, `/reset`, `/history`).

## Project Structure
```
chat_bot/
├─ data/                # intents.json and other data files
├─ models/              # trained model and preprocessor
├─ src/                 # core chatbot engine
│   └─ chatbot.py       # ChatbotEngine implementation
├─ main.py              # CLI entry point
├─ install_deps.py      # Install dependencies from requirements.txt
├─ requirements.txt     # Python package requirements
├─ README.md            # This file
├─ .gitignore           # Git ignored files
└─ LICENSE              # MIT License
```

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/sanT-chatbot.git
cd sanT-chatbot

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install required packages
python install_deps.py
```

## Usage
```bash
python main.py
```
You will see a banner such as:
```
[BOT]  sanT DEEP LEARNING CHATBOT  [BOT]
```
Type your messages and interact with the bot. Use `quit` or `exit` to stop.

## Adding New Intents
Edit `data/intents.json` and add a new object with `tag`, `patterns`, and `responses`. Restart the bot to load the changes.

## Contributing
Feel free to open issues or submit pull requests. Ensure you follow the existing code style and run tests (if added).

## License
This project is licensed under the MIT License – see the `LICENSE` file for details.
#
