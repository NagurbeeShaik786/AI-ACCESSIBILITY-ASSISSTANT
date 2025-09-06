import os
import logging
import io
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import tempfile
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_12345")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logging.info("Rendering index.html (simple_app)")
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "message": "AI Accessibility Assistant Backend",
        "services": {
            "ocr": False,
            "summarization": False,
            "tts": False
        },
        "status": "healthy"
    })

@app.route('/api/image-to-speech', methods=['POST'])
def image_to_speech():
    return jsonify({'error': 'OCR functionality requires additional packages to be installed. Please install pytesseract, Pillow, and gTTS.'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    return jsonify({'error': 'Text-to-speech functionality requires gTTS package to be installed.'}), 500

@app.route('/api/speech-to-sign', methods=['POST'])
def speech_to_sign():
    return jsonify({'error': 'Speech recognition functionality requires SpeechRecognition package to be installed.'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)