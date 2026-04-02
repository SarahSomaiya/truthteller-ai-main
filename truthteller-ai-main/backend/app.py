from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils import allowed_file, extract_text_from_file, analyze_text

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            result = analyze_text(text)
            return jsonify(result), 200
        
        elif 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                file_extension = filename.rsplit('.', 1)[1].lower()
                text = extract_text_from_file(file_path, file_extension)
                
                if not text:
                    return jsonify({'error': 'Text extraction failed'}), 400
                
                result = analyze_text(text)
                return jsonify(result), 200
            
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        else:
            return jsonify({'error': 'Invalid request'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/api', methods=['GET'])
def index():
    return jsonify({
        'name': 'TruthTeller AI Backend',
        'version': '1.0.0'
    }), 200


# REQUIRED FOR VERCEL
def handler(request):
    return app(request)