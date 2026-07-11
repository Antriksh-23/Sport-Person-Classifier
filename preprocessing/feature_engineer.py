import cv2
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler

def w2d(img, mode='haar', level=1):
    """
    Applies the Wavelet Transform to an image.
    Returns the transformed image.
    """
    imArray = img
    # Datatype conversions
    # Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255.
    
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;
    
    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255.
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H

def generate_feature_vector(cropped_image: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Step 4-9 of preprocessing:
    4: Resize images to fixed dimensions.
    5: Generate wavelet-transformed images.
    6: Flatten RGB image.
    7: Flatten wavelet image.
    8: Concatenate both vectors.
    (Normalization is usually done collectively before training, but we can return raw concatenated first, 
    or scale them here if needed).
    
    Args:
        cropped_image: The output of the face_detector.
        size: The fixed width/height for resizing.
        
    Returns:
        Flattened combined feature vector.
    """
    # Step 4: Resize
    scalled_raw_img = cv2.resize(cropped_image, (size, size))
    
    # Step 5: Wavelet transform
    img_har = w2d(cropped_image, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (size, size))
    
    # Step 6 & 7: Flatten RGB and Wavelet images
    # rgb image is size * size * 3, wavelet is size * size
    combined_img = np.vstack((scalled_raw_img.reshape(size * size * 3, 1), 
                              scalled_img_har.reshape(size * size, 1)))
                              
    # Step 8: Concatenate
    flattened_vector = combined_img.flatten()
    
    return flattened_vector
