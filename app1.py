from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging
import os

# Initialize Flask app
app = Flask(__name__)

os.environ["HUGGING_FACE_API_KEY"] = "" # Use your own token

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # Replace with your preferred model

@app.route("/")
def home():
    """Render the main UI."""
    return render_template("index1.html")  # Ensure this file is placed in the `templates` folder

@app.route('/qa', methods=['POST'])
def question_answering():
    """Answer questions based on the submitted document."""
    data = request.json
    text = data.get('text')
    question = data.get('question')

    if not text or not question:
        return jsonify({'error': 'Both document text and a question are required.'}), 400

    try:
        answer = qa_pipeline({'question': question, 'context': text})
        return jsonify({'answer': answer['answer']})
    
        
    except Exception as e:
        logger.error(f"Error during question answering: {str(e)}")
        return jsonify({'error': 'Failed to process the question. Please try again later.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
