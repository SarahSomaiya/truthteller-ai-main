from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils import allowed_file, extract_text_from_file, analyze_text

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle both text and file uploads for analysis
    """
    try:
        # Check if this is a JSON request with text
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text or text.strip() == '':
                return jsonify({
                    'error': 'Could not extract text. This file may be scanned or image-based.'
            }), 400
            
            result = analyze_text(text)
            return jsonify(result), 200
        
        # Check if this is a file upload
        elif 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected'
                }), 400
            
            # Check file extension
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Extract text from file
                file_extension = filename.rsplit('.', 1)[1].lower()
                text = extract_text_from_file(file_path, file_extension)
                
                if not text or text.strip() == '':
                    return jsonify({
                        'error': 'Could not extract text from file'
                    }), 400
                
                result = analyze_text(text)
                return jsonify(result), 200
            
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        else:
            return jsonify({
                'error': 'Invalid request. Provide either JSON text or a file.'
            }), 400
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'TruthTeller AI Backend',
        'version': '1.0.0',
        'endpoints': {
            'predict': 'POST /predict - Analyze text or file',
            'health': 'GET /health - Health check'
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
