import os
import sys

# Add subdirectories to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from dataset_pipeline import prepare_dataset
from train_model import train_and_evaluate

if __name__ == "__main__":
    raw_dataset_path = "./dataset"
    artifacts_path = "./dataset/cropped"
    models_dir = "./models"
    eval_dir = "./evaluation"

    print("Step 1: Preparing dataset and generating features...")
    try:
        X, y, class_dict = prepare_dataset(raw_dataset_path, artifacts_path)
    except FileNotFoundError:
        print(f"Error: Could not find the '{raw_dataset_path}' directory. Please create it and add athlete image folders.")
        sys.exit(1)

    if len(X) == 0:
        print("No valid images found in the dataset. Please ensure you have folders with valid face images inside 'dataset/'.")
        sys.exit(1)

    print("Step 2: Training models and saving artifacts...")
    train_and_evaluate(X, y, class_dict, models_dir=models_dir, eval_dir=eval_dir)
    print("Done! You can now start the Flask server.")
