from flask import Flask, request, jsonify, render_template
import sys
import os

# Add utils path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import predict_utils

# Point Flask to the correct static and template folders
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app = Flask(__name__, 
            template_folder=os.path.join(frontend_dir, 'templates'),
            static_folder=os.path.join(frontend_dir, 'static'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        image_data = data['image_data']
        response = predict_utils.classify_image(image_data)
        
        if "error" in response:
            return jsonify(response), 400
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification...")
    predict_utils.load_saved_artifacts()
    app.run(port=5000, debug=True)
