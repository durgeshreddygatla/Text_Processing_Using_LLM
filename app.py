from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging
import os
import pyttsx3

# Initialize Flask app
app = Flask(__name__)

os.environ["HUGGING_FACE_API_KEY"] = "" # Use your own token

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the text generation model
generator = pipeline("text-generation", model="gpt2")  # Replace with your preferred model

@app.route("/")
def home():
    """Render chatbot UI."""
    return render_template("index.html")  # Create this file in the `templates` folder

@app.route('/text-generation', methods=['POST'])
def generation():
    """Generate text based on the provided prompt."""
    text = request.form.get('text') or request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        result = generator(text, max_length=70, num_return_sequences=1)
        return jsonify({'Generated text': result[0]['generated_text']})
    except Exception as e:
        logger.error(f"Generation Error: {str(e)}")
        return jsonify({'error': 'Text Generation failed. Please try again later.'}), 500



# Initialize the text-to-speech engine
engine = pyttsx3.init()

@app.route('/speak', methods=['POST'])
def speak():
    """Convert text to speech and return the audio."""
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Save the speech to a file
        engine.save_to_file(text, 'response.mp3')
        engine.runAndWait()
        return jsonify({'audio': '/response.mp3'})
    except Exception as e:
        logger.error(f"Speech Generation Error: {str(e)}")
        return jsonify({'error': 'Speech generation failed'}), 500





if __name__ == "__main__":
    app.run(debug=True)

