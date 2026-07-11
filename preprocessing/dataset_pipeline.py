import os
import shutil
import numpy as np
from face_detector import get_cropped_image_if_2_eyes
from feature_engineer import generate_feature_vector

def prepare_dataset(raw_dataset_path: str, artifacts_path: str):
    """
    1. Dataset organization
    2. Image cleaning
    Iterates through the raw dataset, crops faces, and saves them to the artifacts folder.
    Also returns lists of features (X) and labels (y) along with the class dictionary.
    """
    class_dict = {}
    count = 0
    X, y = [], []
    
    # If the cropped folder already exists and has contents, use it directly 
    # to avoid overwriting manually cleaned datasets
    if os.path.exists(artifacts_path) and len(os.listdir(artifacts_path)) > 0:
        print(f"Loading existing cropped images from {artifacts_path}...")
        for item in os.listdir(artifacts_path):
            folder_path = os.path.join(artifacts_path, item)
            if os.path.isdir(folder_path):
                class_name = item
                class_dict[class_name] = count
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    import cv2
                    img = cv2.imread(file_path)
                    if img is not None:
                        features = generate_feature_vector(img)
                        X.append(features)
                        y.append(class_dict[class_name])
                count += 1
        return np.array(X).astype(float), np.array(y), class_dict

    # Otherwise, create it and crop from the raw dataset
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    else:
        shutil.rmtree(artifacts_path)
        os.makedirs(artifacts_path)
    
    # Iterate over subdirectories in the raw dataset
    for item in os.listdir(raw_dataset_path):
        folder_path = os.path.join(raw_dataset_path, item)
        
        if os.path.isdir(folder_path):
            if item == "cropped":
                continue
            class_name = item
            class_dict[class_name] = count
            
            # Create a cropped folder for this class
            cropped_folder_path = os.path.join(artifacts_path, class_name)
            os.makedirs(cropped_folder_path)
            
            file_count = 1
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if it's an image
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # 3. Face detection, 4. Eye verification, 5. Automatic cropping
                roi_color = get_cropped_image_if_2_eyes(file_path)
                
                if roi_color is not None:
                    # Save cropped image (optional but good for debugging)
                    import cv2
                    cropped_file_path = os.path.join(cropped_folder_path, f"{class_name}_{file_count}.png")
                    cv2.imwrite(cropped_file_path, roi_color)
                    file_count += 1
                    
                    # 6. Generate Feature Vector (Resize, Wavelet, Flatten, Concat)
                    features = generate_feature_vector(roi_color)
                    
                    X.append(features)
                    y.append(class_dict[class_name])
                    
            count += 1
            
    # Step 9: Normalize feature vectors will be handled in the training script via StandardScaler
    return np.array(X).astype(float), np.array(y), class_dict
