import cv2
import os
import glob

def augment_dataset(dataset_path='./dataset'):
    print("Starting Data Augmentation...")
    folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    
    total_augmented = 0
    
    for folder in folders:
        person_name = os.path.basename(folder)
        print(f"Processing folder: {person_name}")
        
        # Find all images in the folder
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
            
        count = 0
        for img_path in image_files:
            # Skip images that are already augmented to avoid infinite loops
            if '_aug_' in img_path:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Augmentation 1: Horizontal Flip
            flipped_img = cv2.flip(img, 1)
            flip_path = img_path.rsplit('.', 1)
            flip_path = f"{flip_path[0]}_aug_flip.{flip_path[1]}"
            cv2.imwrite(flip_path, flipped_img)
            
            count += 1
            total_augmented += 1
            
        print(f" -> Generated {count} new augmented images for {person_name}")
        
    print(f"\nDone! Successfully created {total_augmented} new augmented images across all folders.")

if __name__ == "__main__":
    augment_dataset()
