# TruthTeller AI - Flask Backend Setup Guide

## Overview
This Flask backend provides the API for the TruthTeller AI frontend. It handles text analysis and document processing (PDF, DOCX, PPTX).

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Backend

```bash
python app.py
```

The backend will start on `http://localhost:5000`

You should see output like:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://localhost:5000
```

## API Endpoints

### 1. Text Analysis
**Endpoint:** `POST /predict`
**Content-Type:** `application/json`

**Request:**
```json
{
  "text": "Your text to analyze here..."
}
```

**Response:**
```json
{
  "prediction": 75.5,
  "confidence": 0.85
}
```

### 2. File Upload Analysis
**Endpoint:** `POST /predict`
**Content-Type:** `multipart/form-data`

**Supported Formats:** PDF, DOCX, PPTX

**Response:**
```json
{
  "prediction": 75.5,
  "confidence": 0.85
}
```

### 3. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

### 4. API Info
**Endpoint:** `GET /`

## How to Use with Frontend

1. **Start the backend first:**
   ```bash
   python app.py
   ```

2. **In another terminal, start the frontend:**
   ```bash
   npm run dev
   ```

3. The frontend will automatically connect to `http://localhost:5000/predict`

## Current Analysis Algorithm

The backend uses a heuristic-based analysis that:
- Identifies suspicious language patterns
- Calculates credibility scores (0-100)
- Provides confidence levels based on text length

**For production use:** Replace the `analyze_text()` function in `utils.py` with your ML model.

## Troubleshooting

### Port 5000 Already in Use
```bash
# Find and kill process using port 5000
# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### ImportError with Dependencies
Make sure you're in the virtual environment:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### Uploads Folder Not Found
The `uploads/` folder is created automatically when you first upload a file.

## Next Steps

1. Implement your own ML model in `analyze_text()` function
2. Add database integration for results storage
3. Implement user authentication
4. Add logging and monitoring
