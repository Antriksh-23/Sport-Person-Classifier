import os
import random

def balance_dataset(dataset_dir="dataset/cropped", max_images=80):
    """
    Balances the dataset by ensuring no folder has more than `max_images`.
    Randomly deletes excess images to prevent class imbalance.
    """
    print(f"Scanning '{dataset_dir}' to balance classes to a maximum of {max_images} images...\n")
    
    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) > max_images:
            # Randomly select images to delete
            images_to_delete = random.sample(images, len(images) - max_images)
            for img in images_to_delete:
                img_path = os.path.join(folder_path, img)
                os.remove(img_path)
                
            print(f"✅ Balanced '{folder_name}': Removed {len(images_to_delete)} images -> (Now has {max_images})")
        else:
            print(f"⏭️ Skipped '{folder_name}': Only has {len(images)} images (No balancing needed)")

if __name__ == "__main__":
    balance_dataset()
    print("\nDone! Your dataset is now balanced. You can retrain your model!")
