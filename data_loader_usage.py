import os
import sys
import pandas as pd

# Ensure the parent directory is in sys.path for import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_labels, create_balanced_dataframe, split_data, get_image_path, load_and_preprocess_image

# Example usage script for data_loader.py

data_path = '../datasets/CXR8/'  # Adjust path if needed
images_dir = os.path.join(data_path, 'images', 'extracted_images')
labels_df = load_labels(data_path)
balanced_df = create_balanced_dataframe(labels_df)
train_df, val_df, test_df = split_data(balanced_df)

# Ensure train_df is a DataFrame
if isinstance(train_df, pd.DataFrame):
    # Load a sample image
    sample_image_id = train_df.iloc[0]['Image Index']
    sample_image_path = get_image_path(sample_image_id, images_dir)
    img_array = load_and_preprocess_image(sample_image_path)
    if img_array is not None:
        print(f"Loaded image shape: {img_array.shape}")
    else:
        print("Failed to load sample image.")

    # Example: Load and preprocess all images in train_df with progress logging
    print("Loading and preprocessing all training images...")
    train_images = []
    for i, (_, row) in enumerate(train_df.iterrows()):
        img_path = get_image_path(row['Image Index'], images_dir)
        img = load_and_preprocess_image(img_path)
        if img is not None:
            train_images.append(img)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(train_df)} images")
    print(f"Finished loading {len(train_images)} training images.")
else:
    print("train_df is not a pandas DataFrame. Please check the data_loader functions.") 