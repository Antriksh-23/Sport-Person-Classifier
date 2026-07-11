import os
import joblib
import json
import base64
import cv2
import numpy as np
import sys

# Add preprocessing path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from face_detector import get_cropped_image_if_2_eyes
from feature_engineer import generate_feature_vector

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model
    
    models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    try:
        with open(os.path.join(models_path, "class_dictionary.json"), "r") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

        if __model is None:
            with open(os.path.join(models_path, "saved_model.pkl"), 'rb') as f:
                __model = joblib.load(f)
        print("Loading saved artifacts...done")
    except FileNotFoundError:
        print("WARNING: Model artifacts not found. Please add images to 'dataset/' and run the training pipeline first.")

def get_cv2_image_from_base64_string(b64str):
    '''
    Convert base64 string to opencv image format.
    '''
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def class_number_to_name(class_num):
    return __class_number_to_name.get(class_num, "Unknown")

def classify_image(image_base64_data, file_path=None):
    if not __model:
        load_saved_artifacts()

    # Preprocessing
    if image_base64_data:
        img = get_cv2_image_from_base64_string(image_base64_data)
        if img is None:
            raise ValueError("Invalid base64 image data")
        
        # Save temp for cascade testing if needed, or modify get_cropped... to accept raw img
        temp_path = "temp_predict_img.png"
        cv2.imwrite(temp_path, img)
        roi = get_cropped_image_if_2_eyes(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        roi = get_cropped_image_if_2_eyes(file_path)

    if roi is None:
        return {"error": "No face with two eyes distinctly detected in the image."}

    # Feature Engineering
    features = generate_feature_vector(roi).reshape(1, -1)
    
    # Model inference
    try:
        class_num = __model.predict(features)[0]
        prob_array = np.round(__model.predict_proba(features) * 100, 2).tolist()[0]
        
        probabilities = {}
        for class_name, cls_num in __class_name_to_number.items():
            probabilities[class_name] = prob_array[cls_num]
            
        predicted_name = class_number_to_name(class_num)
        confidence = prob_array[class_num]
        
        # Confidence Threshold: If less than 40%, override to Unknown
        if confidence < 40.0:
            predicted_name = "Unknown person"
            
        return {
            "athlete": predicted_name,
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    load_saved_artifacts()
