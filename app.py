from flask import Flask, request, jsonify, render_template, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Define the path to your local model files
model_path = "gpt2-finetuned/checkpoint-3"

# Load the model and tokenizer with error handling
try:
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt.strip():
        logging.warning("Empty prompt received.")
        return jsonify({"error": "Prompt is required"}), 400

    if len(prompt) > 1000:
        logging.warning("Prompt is too long.")
        return jsonify({"error": "Prompt is too long"}), 400

    logging.info(f"User prompt: {prompt}")

    # Retrieve conversation history from the session
    if "history" not in session:
        session["history"] = []

    session["history"].append({"role": "user", "message": prompt})

    # Generate a response using the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    logging.info(f"Bot response: {response}")

    session["history"].append({"role": "bot", "message": response})

    return jsonify({"response": response, "history": session["history"]})

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": session.get("history", [])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)