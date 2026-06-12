from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load model
print("Loading model...")
if os.path.exists('model.pkl'):
    try:
        model = joblib.load('model.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ model.pkl not found! Run 'python train_model.py' first")
    model = None

def clean_text(text):
    """Bersihkan teks dari karakter aneh"""
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        news_text = data.get('news_text', '')
        
        if not news_text or len(news_text.strip()) < 10:
            return jsonify({'error': 'Teks berita terlalu pendek (minimal 10 karakter)'}), 400
        
        cleaned_text = clean_text(news_text)
        
        if model is None:
            return jsonify({'error': 'Model belum siap'}), 500
            
        # Prediksi
        prediction = model.predict([cleaned_text])[0]
        confidence = model.predict_proba([cleaned_text])[0]
        confidence_score = float(max(confidence) * 100)  # <-- UBAH KE FLOAT
        is_fake = bool(prediction == 0)  # <-- UBAH KE BOOL PYTHON
        
        result = "Real News" if prediction == 1 else "Fake News"
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence_score, 2),
            'is_fake': is_fake,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)