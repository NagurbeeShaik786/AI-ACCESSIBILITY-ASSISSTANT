import os
import logging
import io
import base64
import json
import time
import tempfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
try:
    from pydub import AudioSegment
    from pydub.utils import which
except ImportError:
    AudioSegment = None
try:
    import pyaudio
except ImportError:
    pyaudio = None
try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = None
    KaldiRecognizer = None
try:
    import ollama
except ImportError:
    ollama = None
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import requests
except ImportError:
    requests = None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_12345")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = os.path.abspath('static/uploads')
AUDIO_FOLDER = os.path.abspath('static/audio')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", r"D:\aiii\vosk-model-small-en-us-0.15")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Set FFmpeg and ffprobe paths for pydub
if AudioSegment:
    AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe = which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"
    if not os.path.exists(AudioSegment.converter):
        logging.error(f"ffmpeg not found at {AudioSegment.converter}")
    if not os.path.exists(AudioSegment.ffprobe):
        logging.error(f"ffprobe not found at {AudioSegment.ffprobe}")

# Set Tesseract path explicitly for Windows
if pytesseract:
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        logging.info(f"Tesseract path set to {pytesseract.pytesseract.tesseract_cmd}")
    except Exception as e:
        logging.error(f"Error setting Tesseract path: {e}")
        pytesseract = None

# Initialize AI models
summarizer = None
asr = None
vosk_model = None
if pipeline:
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        logging.info("Summarization model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load summarization model: {e}")
    try:
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        logging.info("ASR model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load ASR model: {e}")
if Model and os.path.exists(VOSK_MODEL_PATH):
    try:
        vosk_model = Model(VOSK_MODEL_PATH)
        logging.info("Vosk model initialized globally")
    except Exception as e:
        logging.error(f"Failed to load Vosk model: {e}")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_audio_file(filepath):
    """Check if audio file is valid and non-empty."""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    if os.path.getsize(filepath) < 100:  # Check for near-empty files
        return False, "File is empty or too small"
    return True, None

def create_tts_with_timeout(text, lang='en', timeout=10):
    """Create gTTS with retry and timeout handling."""
    if gTTS and requests:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.session = session
        return tts
    return None

def text_to_speech_pyttsx3(text, language='en'):
    """Fallback TTS using pyttsx3."""
    if not pyttsx3:
        raise Exception("pyttsx3 not available")
    engine = pyttsx3.init()
    engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Alex' if language == 'en' else 'default')
    audio_buffer = io.BytesIO()
    temp_file = os.path.join(AUDIO_FOLDER, "temp_tts.wav")
    engine.save_to_file(text, temp_file)
    engine.runAndWait()
    with open(temp_file, 'rb') as f:
        audio_buffer.write(f.read())
    audio_buffer.seek(0)
    os.remove(temp_file)
    return audio_buffer

@app.route('/')
def index():
    logging.info("Rendering index.html")
    return render_template('index.html')

@app.route('/health')
def health():
    ocr_available = bool(Image and pytesseract)
    tts_available = bool(gTTS or pyttsx3)
    stt_available = bool(asr or (sr and AudioSegment) or (Model and KaldiRecognizer and pyaudio and vosk_model))
    summarization_available = bool(summarizer)
    ffmpeg_available = bool(AudioSegment and os.path.exists(AudioSegment.converter) and os.path.exists(AudioSegment.ffprobe))
    ollama_available = bool(ollama)
    return jsonify({
        "message": "AI Accessibility Assistant Backend",
        "services": {
            "ocr": ocr_available,
            "speech_to_text": stt_available,
            "text_to_speech": tts_available,
            "summarization": summarization_available,
            "ffmpeg": ffmpeg_available,
            "ollama": ollama_available
        },
        "status": "healthy" if stt_available and ffmpeg_available else "degraded"
    })

@app.route('/api/image-to-speech', methods=['POST'])
def image_to_speech():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_IMAGE_EXTENSIONS}'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Image saved to {filepath}")
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save image file'}), 500
        
        if not Image:
            return jsonify({'error': 'OCR functionality not available - missing Pillow'}), 500
        if not pytesseract:
            return jsonify({'error': 'OCR functionality not available - Tesseract not installed or not in PATH'}), 500
        
        try:
            image = Image.open(filepath).convert('RGB')
            text = pytesseract.image_to_string(image)
            logging.info(f"Extracted text: {text[:100]}...")
        except Exception as e:
            return jsonify({'error': f'OCR failed: {str(e)}'}), 500
        
        if not text.strip():
            return jsonify({'error': 'No text found in image'}), 400
        
        language = request.form.get('language', 'en')
        audio_buffer = None
        if gTTS:
            try:
                tts = create_tts_with_timeout(text, language, timeout=10)
                if tts:
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
            except Exception as e:
                logging.error(f"gTTS failed: {str(e)}. Falling back to pyttsx3.")
        
        if not audio_buffer and pyttsx3:
            try:
                audio_buffer = text_to_speech_pyttsx3(text, language)
            except Exception as e:
                logging.error(f"pyttsx3 failed: {str(e)}")
                return jsonify({'error': f'TTS failed: {str(e)}'}), 500
        
        if not audio_buffer:
            return jsonify({'error': 'TTS functionality not available - missing gTTS and pyttsx3'}), 500
        
        audio_data = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            'text': text,
            'audio_data': audio_data,
            'message': 'Text extracted and converted to speech successfully'
        })
        
    except Exception as e:
        logging.error(f"Image to speech error: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logging.error(f"Failed to delete file {filepath}: {e}")

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided in JSON body'}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text is empty'}), 400
        
        language = data.get('language', 'en')
        summarize = data.get('summarize', False)
        
        processed_text = text
        
        if summarize and summarizer and len(text) > 100:
            try:
                max_length = 1024
                if len(text) > max_length:
                    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                    summaries = []
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:
                            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                            summaries.append(summary[0]['summary_text'])
                    processed_text = ' '.join(summaries)
                else:
                    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
                    processed_text = summary[0]['summary_text']
                logging.info(f"Summarized text: {processed_text[:100]}...")
            except Exception as e:
                logging.error(f"Summarization error: {e}")
                processed_text = text
        
        audio_buffer = None
        if gTTS:
            try:
                tts = create_tts_with_timeout(processed_text, language, timeout=10)
                if tts:
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
            except Exception as e:
                logging.error(f"gTTS failed: {str(e)}. Falling back to pyttsx3.")
        
        if not audio_buffer and pyttsx3:
            try:
                audio_buffer = text_to_speech_pyttsx3(processed_text, language)
            except Exception as e:
                logging.error(f"pyttsx3 failed: {str(e)}")
                return jsonify({'error': f'TTS failed: {str(e)}'}), 500
        
        if not audio_buffer:
            return jsonify({'error': 'TTS functionality not available - missing gTTS and pyttsx3'}), 500
        
        audio_data = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            'original_text': text,
            'processed_text': processed_text,
            'audio_data': audio_data,
            'was_summarized': summarize and processed_text != text,
            'message': 'Text converted to speech successfully'
        })
        
    except Exception as e:
        logging.error(f"Text to speech error: {e}", exc_info=True)
        return jsonify({'error': f'Failed to convert text to speech: {str(e)}'}), 500

@app.route('/api/speech-to-sign', methods=['POST'])
def speech_to_sign():
    temp_audio_path = None
    converted_audio_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if not asr and not (Model and KaldiRecognizer and vosk_model) and not (sr and AudioSegment):
            return jsonify({'error': 'Speech recognition not available - missing required dependencies'}), 500
        
        if not AudioSegment or not os.path.exists(AudioSegment.converter) or not os.path.exists(AudioSegment.ffprobe):
            return jsonify({'error': 'FFmpeg is required for audio processing but is not installed'}), 500
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': f'Invalid audio type. Allowed: {ALLOWED_AUDIO_EXTENSIONS}'}), 400
        
        filename = secure_filename(audio_file.filename)
        temp_audio_path = os.path.join(AUDIO_FOLDER, filename)
        audio_file.save(temp_audio_path)
        logging.info(f"Audio saved to {temp_audio_path}")
        
        is_valid, error_msg = validate_audio_file(temp_audio_path)
        if not is_valid:
            return jsonify({'error': f'Invalid audio file: {error_msg}'}), 400
        
        input_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        converted_audio_path = temp_audio_path
        if input_ext != 'wav':
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                converted_audio_path = os.path.join(AUDIO_FOLDER, f"converted_{filename.rsplit('.', 1)[0]}.wav")
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(converted_audio_path, format='wav')
                logging.info(f"Converted audio to {converted_audio_path}")
            except Exception as e:
                logging.error(f"Audio conversion failed: {str(e)}")
                return jsonify({'error': f'Audio conversion failed: {str(e)}'}), 500
        
        text = None
        recognition_method = None
        
        if asr:
            try:
                result = asr(converted_audio_path)
                text = result['text'].strip()
                recognition_method = "Whisper"
                logging.info(f"Recognized text (Whisper): {text}")
            except Exception as e:
                logging.error(f"Whisper recognition failed: {str(e)}")
        
        if text is None and Model and KaldiRecognizer and vosk_model:
            try:
                rec = KaldiRecognizer(vosk_model, 16000)
                with open(converted_audio_path, "rb") as f:
                    while True:
                        data = f.read(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get('text', '').strip()
                            if text:
                                recognition_method = "Vosk"
                                break
                if not text:
                    result = json.loads(rec.FinalResult())
                    text = result.get('text', '').strip()
                    recognition_method = "Vosk"
                logging.info(f"Recognized text (Vosk): {text}")
            except Exception as e:
                logging.error(f"Vosk recognition failed: {str(e)}")
        
        if text is None and sr and AudioSegment:
            try:
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                with sr.AudioFile(converted_audio_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language='en-US')
                    recognition_method = "Google"
                    logging.info(f"Recognized text (Google): {text}")
                except sr.UnknownValueError:
                    logging.warning("Google could not understand audio")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
                except sr.RequestError as e:
                    logging.error(f"Google service error: {e}")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
            except Exception as e:
                logging.error(f"Speech recognition failed: {str(e)}")
        
        if not text:
            return jsonify({'error': 'Could not recognize speech. Ensure audio is clear and contains speech.'}), 400
        
        translated_text = text
        target_language = request.form.get('target_language', 'English')
        if ollama:
            try:
                response = ollama.generate(
                    model="llama3",
                    prompt=f"Translate the following text to {target_language}: {text}"
                )
                translated_text = response['response'].strip()
                logging.info(f"Translated text (Ollama): {translated_text}")
            except Exception as e:
                logging.error(f"Ollama translation failed: {str(e)}. Falling back to original text.")
        
        sign_data = text_to_sign_language(translated_text)
        
        return jsonify({
            'recognized_text': text,
            'translated_text': translated_text,
            'recognition_method': recognition_method,
            'sign_data': sign_data,
            'message': 'Speech recognized, translated, and converted to sign language'
        })
        
    except Exception as e:
        logging.error(f"Audio processing error: {e}", exc_info=True)
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
    finally:
        for path in [temp_audio_path, converted_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logging.info(f"Cleaned up file: {path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {path}: {e}")

@app.route('/api/microphone-to-sign', methods=['POST'])
def microphone_to_sign():
    temp_audio_path = None
    converted_audio_path = None
    try:
        if not asr and not (Model and KaldiRecognizer and vosk_model) and not (sr and AudioSegment):
            return jsonify({'error': 'Speech recognition not available - missing required dependencies'}), 500
        
        if not AudioSegment or not os.path.exists(AudioSegment.converter) or not os.path.exists(AudioSegment.ffprobe):
            return jsonify({'error': 'FFmpeg is required for audio processing but is not installed'}), 500
        
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio data provided in JSON body'}), 400
        
        audio_base64 = data.get('audio', '').strip()
        if not audio_base64:
            return jsonify({'error': 'Audio data is empty'}), 400
        
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            logging.error(f"Base64 decoding failed: {str(e)}")
            return jsonify({'error': f'Invalid audio data: {str(e)}'}), 400
        
        with tempfile.NamedTemporaryFile(suffix='.wav', dir=AUDIO_FOLDER, delete=False) as temp_file:
            temp_audio_path = temp_file.name
            temp_file.write(audio_data)
            logging.info(f"Microphone audio saved to {temp_audio_path}")
        
        is_valid, error_msg = validate_audio_file(temp_audio_path)
        if not is_valid:
            return jsonify({'error': f'Invalid audio file: {error_msg}'}), 400
        
        converted_audio_path = temp_audio_path
        try:
            audio = AudioSegment.from_file(temp_audio_path)
            converted_audio_path = os.path.join(AUDIO_FOLDER, f"converted_{os.path.basename(temp_audio_path)}")
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(converted_audio_path, format='wav')
            logging.info(f"Converted audio to {converted_audio_path}")
        except Exception as e:
            logging.error(f"Audio conversion failed: {str(e)}")
            return jsonify({'error': f'Audio conversion failed: {str(e)}'}), 500
        
        text = None
        recognition_method = None
        
        if asr:
            try:
                result = asr(converted_audio_path)
                text = result['text'].strip()
                recognition_method = "Whisper"
                logging.info(f"Recognized text (Whisper): {text}")
            except Exception as e:
                logging.error(f"Whisper recognition failed: {str(e)}")
        
        if text is None and Model and KaldiRecognizer and vosk_model:
            try:
                rec = KaldiRecognizer(vosk_model, 16000)
                with open(converted_audio_path, "rb") as f:
                    while True:
                        data = f.read(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get('text', '').strip()
                            if text:
                                recognition_method = "Vosk"
                                break
                if not text:
                    result = json.loads(rec.FinalResult())
                    text = result.get('text', '').strip()
                    recognition_method = "Vosk"
                logging.info(f"Recognized text (Vosk): {text}")
            except Exception as e:
                logging.error(f"Vosk recognition failed: {str(e)}")
        
        if text is None and sr and AudioSegment:
            try:
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                with sr.AudioFile(converted_audio_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language='en-US')
                    recognition_method = "Google"
                    logging.info(f"Recognized text (Google): {text}")
                except sr.UnknownValueError:
                    logging.warning("Google could not understand audio")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
                except sr.RequestError as e:
                    logging.error(f"Google service error: {e}")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
            except Exception as e:
                logging.error(f"Speech recognition failed: {str(e)}")
        
        if not text:
            return jsonify({'error': 'Could not recognize speech. Ensure audio is clear and contains speech.'}), 400
        
        translated_text = text
        target_language = data.get('target_language', 'English')
        if ollama:
            try:
                response = ollama.generate(
                    model="llama3",
                    prompt=f"Translate the following text to {target_language}: {text}"
                )
                translated_text = response['response'].strip()
                logging.info(f"Translated text (Ollama): {translated_text}")
            except Exception as e:
                logging.error(f"Ollama translation failed: {str(e)}. Falling back to original text.")
        
        sign_data = text_to_sign_language(translated_text)
        
        return jsonify({
            'recognized_text': text,
            'translated_text': translated_text,
            'recognition_method': recognition_method,
            'sign_data': sign_data,
            'message': 'Microphone speech recognized, translated, and converted to sign language'
        })
        
    except Exception as e:
        logging.error(f"Microphone audio processing error: {e}", exc_info=True)
        return jsonify({'error': f'Failed to process microphone audio: {str(e)}'}), 500
    finally:
        for path in [temp_audio_path, converted_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logging.info(f"Cleaned up file: {path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {path}: {e}")

@app.route('/api/listen-to-text', methods=['POST'])
def listen_to_text():
    temp_audio_path = None
    try:
        if not (Model and KaldiRecognizer and pyaudio and vosk_model) and not sr:
            return jsonify({'error': 'Live speech recognition not available - missing required dependencies'}), 500
        
        data = request.get_json()
        duration = data.get('duration', 10) if data else 10
        language = data.get('language', 'en-US') if data else 'en-US'
        
        text = None
        recognition_method = None
        
        if Model and KaldiRecognizer and pyaudio and vosk_model:
            try:
                rec = KaldiRecognizer(vosk_model, 16000)
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
                stream.start_stream()
                logging.info("Listening with Vosk...")
                start_time = time.time()
                full_text = ""
                while time.time() - start_time < duration:
                    data = stream.read(4000, exception_on_overflow=False)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        full_text += result.get('text', '') + " "
                    else:
                        partial = json.loads(rec.PartialResult())
                        if partial.get('partial'):
                            logging.info(f"Partial: {partial['partial']}")
                result = json.loads(rec.FinalResult())
                full_text += result.get('text', '')
                text = full_text.strip()
                recognition_method = "Vosk"
                stream.stop_stream()
                stream.close()
                p.terminate()
                logging.info(f"Recognized text (Vosk): {text}")
            except Exception as e:
                logging.error(f"Vosk live recognition failed: {e}")
        
        if text is None and sr:
            try:
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    logging.info("Listening with speech_recognition...")
                    audio_data = r.listen(source, timeout=duration, phrase_time_limit=duration)
                try:
                    text = r.recognize_google(audio_data, language=language)
                    recognition_method = "Google"
                    logging.info(f"Recognized text (Google): {text}")
                except sr.UnknownValueError:
                    logging.warning("Google could not understand audio")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
                except sr.RequestError as e:
                    logging.error(f"Google service error: {e}")
                    text = r.recognize_sphinx(audio_data)
                    recognition_method = "Sphinx"
                    logging.info(f"Recognized text (Sphinx): {text}")
            except Exception as e:
                logging.error(f"Speech recognition failed: {e}")
        
        if not text:
            return jsonify({'error': 'Could not recognize speech. Ensure microphone is working and speech is clear.'}), 400
        
        translated_text = text
        target_language = data.get('target_language', 'English')
        if ollama:
            try:
                response = ollama.generate(
                    model="llama3",
                    prompt=f"Translate the following text to {target_language}: {text}"
                )
                translated_text = response['response'].strip()
                logging.info(f"Translated text (Ollama): {translated_text}")
            except Exception as e:
                logging.error(f"Ollama translation failed: {str(e)}. Falling back to original text.")
        
        return jsonify({
            'recognized_text': text,
            'translated_text': translated_text,
            'recognition_method': recognition_method,
            'message': 'Speech recognized and translated successfully'
        })
    
    except Exception as e:
        logging.error(f"Live audio processing error: {e}", exc_info=True)
        return jsonify({'error': f'Failed to process live audio: {str(e)}'}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                logging.info(f"Cleaned up file: {temp_audio_path}")
            except Exception as e:
                logging.error(f"Failed to delete file {temp_audio_path}: {e}")

def text_to_sign_language(text):
    sign_mapping = {
        'hello': {'gesture': 'wave_hand', 'description': 'Wave hand in greeting'},
        'hi': {'gesture': 'wave_hand', 'description': 'Wave hand in greeting'},
        'goodbye': {'gesture': 'wave_goodbye', 'description': 'Wave hand goodbye'},
        'bye': {'gesture': 'wave_goodbye', 'description': 'Wave hand goodbye'},
        'thank you': {'gesture': 'touch_chin_forward', 'description': 'Touch chin and move hand forward'},
        'thanks': {'gesture': 'touch_chin_forward', 'description': 'Touch chin and move hand forward'},
        'please': {'gesture': 'circle_chest', 'description': 'Make circular motion on chest'},
        'yes': {'gesture': 'nod_fist', 'description': 'Nod fist up and down'},
        'no': {'gesture': 'shake_fingers', 'description': 'Shake index and middle fingers'},
        'help': {'gesture': 'lift_hand', 'description': 'Place one hand on other and lift'},
        'sorry': {'gesture': 'circle_chest_sorry', 'description': 'Make circular motion on chest with sorry expression'},
        'love': {'gesture': 'cross_arms', 'description': 'Cross arms over chest'},
        'family': {'gesture': 'f_shape_circle', 'description': 'Make F shape and circle both hands'},
        'friend': {'gesture': 'hook_fingers', 'description': 'Hook index fingers together'},
        'how': {'gesture': 'back_to_back_hands', 'description': 'Place hands back to back and rotate'},
        'what': {'gesture': 'index_finger_across', 'description': 'Move index finger across other fingers'},
        'where': {'gesture': 'point_shake', 'description': 'Point index finger and shake'},
        'when': {'gesture': 'circle_point', 'description': 'Make circle with one finger around other'},
        'why': {'gesture': 'touch_forehead', 'description': 'Touch forehead with middle finger'},
        'who': {'gesture': 'point_lips', 'description': 'Point to lips with index finger'},
    }
    
    words = text.lower().split()
    sign_sequence = []
    
    for word in words:
        if word in sign_mapping:
            sign_sequence.append(sign_mapping[word])
        else:
            sign_sequence.append({
                'gesture': 'fingerspell',
                'description': f'Fingerspell "{word}"',
                'word': word
            })
    
    return {
        'sequence': sign_sequence,
        'total_signs': len(sign_sequence),
        'estimated_duration': len(sign_sequence) * 2
    }

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logging.info(f"Dependencies - Pillow: {bool(Image)}, pytesseract: {bool(pytesseract)}, "
                 f"gTTS: {bool(gTTS)}, pyttsx3: {bool(pyttsx3)}, speech_recognition: {bool(sr)}, "
                 f"pydub: {bool(AudioSegment)}, summarizer: {bool(summarizer)}, asr: {bool(asr)}, "
                 f"pyaudio: {bool(pyaudio)}, vosk: {bool(Model and KaldiRecognizer and vosk_model)}, "
                 f"ollama: {bool(ollama)}, "
                 f"ffmpeg: {bool(AudioSegment and os.path.exists(AudioSegment.converter) and os.path.exists(AudioSegment.ffprobe))}")
    app.run(host='0.0.0.0', port=5000, debug=True)