import os
import math
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse

app = Flask(__name__)

# 1. Load the Model and Feature Names
# Ensure these files are in your main GitHub folder
MODEL_PATH = 'chinese_english_rf_model_v2.pkl'
FEATURES_PATH = 'chinese_english_feature_names_v2.pkl'

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# 2. Feature Extraction Logic
# This calculates the 20 specific features your model expects
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Logic for features identified in your .pkl metadata
    features = {
        'url_length': len(url),
        'url_entropy': -sum((url.count(c)/len(url)) * math.log2(url.count(c)/len(url)) for c in set(url)),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_at': url.count('@'),
        'is_punycode': 1 if domain.startswith('xn--') else 0,
        'has_non_ascii': 1 if any(ord(c) > 127 for c in url) else 0,
        'is_ip_address': 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0,
        'has_suspicious_keyword': 1 if any(k in url.lower() for k in ['login', 'verify', 'bank', 'secure', 'signin']) else 0,
        'suspicious_keyword_count': sum(url.lower().count(k) for k in ['login', 'verify', 'bank', 'secure', 'signin']),
        'domain_age_days': 365,      # Default placeholder
        'has_dns_record': 1,         # Default placeholder
        'has_iframe': 0,             # Default placeholder
        'path_depth': path.count('/'),
        'param_count': len(parsed_url.query.split('&')) if parsed_url.query else 0,
        'subdomain_count': max(0, domain.count('.') - 1),
        'is_https': 1 if parsed_url.scheme == 'https' else 0,
        'domain_legit_score': 50,    # Default placeholder
        'suspicion_score': 0,        # Default placeholder
        'very_long_url': 1 if len(url) > 75 else 0
    }
    
    # Align features with the order in feature_names_v2.pkl
    return [features[name] for name in feature_names]

# 3. Routes
@app.route('/')
def home():
    # This looks for the HTML file inside a folder named 'templates'
    return render_template('multilingual_phishing_detection_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Extract features and convert to format for Random Forest
        query_features = np.array(extract_features(url)).reshape(1, -1)
        
        # Prediction (0 for Legitimate, 1 for Phishing - or as per your model)
        prediction = model.predict(query_features)[0]
        probability = model.predict_proba(query_features)[0]
        
        # Result mapping
        res_status = "Phishing" if prediction == 1 or prediction == 'phishing' else "Legitimate"
        
        return jsonify({
            'url': url,
            'status': res_status,
            'confidence': f"{round(max(probability) * 100, 2)}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Railway Deployment Port Logic
if __name__ == '__main__':
    # Railway assigns a port dynamically. This code captures it.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
