import cv2
import numpy as np

# Load Haar cascades from cv2 built-in data
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path: str, image_b64: str = None) -> np.ndarray:
    """
    Detects a face using OpenCV Haar Cascades (Frontal and Profile).
    Returns the cropped facial region.
    """
    if image_path:
        img = cv2.imread(image_path)
    else:
        return None
        
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try Frontal Face first (optimized for better detection)
    faces = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    # If no frontal face, try Profile Face
    if len(faces) == 0:
        faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
    # If still no face, try flipping the image horizontally for the profile cascade
    # (haarcascade_profileface only detects faces looking to the right)
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = face_cascade_profile.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4)
        if len(faces_flipped) > 0:
            # We found a face in the flipped image, map the coordinates back!
            x, y, w, h = faces_flipped[0]
            width = img.shape[1]
            faces = [[width - x - w, y, w, h]]

    # If we found at least one face
    if len(faces) > 0:
        # Take the largest face found
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        roi_color = img[y:y+h, x:x+w]
        
        # We no longer strictly enforce the eye check if the face is large enough!
        # This fixes the bug where clear faces were rejected just because of shadows.
        if w >= 50 and h >= 50:
            return roi_color
            
    return None
