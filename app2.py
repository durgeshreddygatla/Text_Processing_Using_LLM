from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

os.environ["HUGGING_FACE_API_KEY"] = "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"
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the image captioning pipeline
try:
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    logger.info("Image captioning pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load image captioning model: {e}")
    image_to_text = None

# Route to render the HTML UI
@app.route("/")
def home():
    """Render the upload page."""
    return render_template("index2.html")

# Route to handle image captioning
@app.route("/caption", methods=["POST"])
def caption_image():
    """Generate a caption for the uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    
    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image file selected.'}), 400

    try:
        # Open the image file
        image = Image.open(image_file)
        
        # Validate the image format
        if image.format not in ["JPEG", "PNG", "BMP", "GIF"]:
            return jsonify({'error': 'Unsupported image format. Please upload JPEG, PNG, BMP, or GIF.'}), 400
        
        # Generate a caption for the image
        result = image_to_text(image)
        caption = result[0]['generated_text'] if result else "No caption could be generated."
        
        return jsonify({'caption': caption})
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return jsonify({'error': 'Failed to generate a caption. Please try again later.'}), 500


if __name__ == "__main__":
    app.run(debug=True)
