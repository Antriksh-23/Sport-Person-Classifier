import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from feature_engineer import generate_feature_vector

def test_generate_feature_vector():
    # Create a dummy image 100x100x3
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Process
    features = generate_feature_vector(dummy_img, size=32)
    
    # Expected size: 32*32*3 (RGB) + 32*32 (Wavelet) = 3072 + 1024 = 4096
    assert features.shape == (4096,)
