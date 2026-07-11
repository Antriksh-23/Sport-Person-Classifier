import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
try:
    import cv2
    print(f"cv2.__file__: {getattr(cv2, '__file__', 'No __file__ attribute')}")
    print(f"cv2.__version__: {getattr(cv2, '__version__', 'No __version__ attribute')}")
    print("Does cv2 have CascadeClassifier?", hasattr(cv2, 'CascadeClassifier'))
except Exception as e:
    print(f"Error importing cv2: {e}")
