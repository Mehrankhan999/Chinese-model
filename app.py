from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import math
import re
from urllib.parse import urlparse

app = Flask(__name__)

# 1. Load the Model and Feature Names
# These must be in the same folder as app.py
model = joblib.load('chinese_english_rf_model_v2.pkl')
feature_names = joblib.load('chinese_english_feature_names_v2.pkl')

# 2. Feature Extraction Logic
# This function calculates the 20 features your model was trained on
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Simple calculation for features identified in your model metadata
    features = {
        'url_length': len(url),
        'url_entropy': -sum((count/len(url)) * math.log2(count/len(url)) for count in [url.count(c) for c in set(url)]),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_at': url.count('@'),
        'is_punycode': 1 if domain.startswith('xn--') else 0,
        'has_non_ascii': 1 if any(ord(c) > 127 for c in url) else 0,
        'is_ip_address': 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0,
        'has_suspicious_keyword': 1 if any(k in url.lower() for k in ['login', 'verify', 'bank', 'secure']) else 0,
        'suspicious_keyword_count': sum(url.lower().count(k) for k in ['login', 'verify', 'bank', 'secure']),
        'domain_age_days': 365,  # Placeholder: requires an external WHOIS API
        'has_dns_record': 1,     # Placeholder
        'has_iframe': 0,         # Placeholder
        'path_depth': path.count('/'),
        'param_count': len(parsed_url.query.split('&')) if parsed_url.query else 0,
        'subdomain_count': max(0, domain.count('.') - 1),
        'is_https': 1 if parsed_url.scheme == 'https' else 0,
        'domain_legit_score': 50, # Placeholder logic
        'suspicion_score': 0,     # Initialized score
        'very_long_url': 1 if len(url) > 75 else 0
    }
    
    # Ensure features are in the exact order the model expects
    return [features[name] for name in feature_names]

# 3. Routes
@app.route('/')
def home():
    return render_template('multilingual_phishing_detection_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL from the dashboard
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Extract features and format for model
        query_features = np.array(extract_features(url)).reshape(1, -1)
        
        # Get Prediction
        prediction = model.predict(query_features)[0]
        probability = model.predict_proba(query_features)[0]
        
        # Return results to the dashboard
        return jsonify({
            'url': url,
            'status': 'Phishing' if prediction == 'phishing' else 'Legitimate',
            'confidence': round(max(probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Railway/Cloud usually provides a PORT variable
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
