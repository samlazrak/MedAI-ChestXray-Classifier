# If running as a standalone script, ensure dependencies are installed:
# pip install numpy pandas pillow scikit-learn

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def load_labels(data_path):
    """
    Load the main labels CSV and add a binary label column (0=Normal, 1=Abnormal).
    """
    print(f"Loading labels from {os.path.join(data_path, 'Data_Entry_2017_v2020.csv')}")
    labels_df = pd.read_csv(os.path.join(data_path, 'Data_Entry_2017_v2020.csv'))
    print(f"Loaded {len(labels_df)} label entries.")
    labels_df['binary_label'] = (labels_df['Finding Labels'] != 'No Finding').astype(int)
    print(f"Added 'binary_label' column. Normal: {(labels_df['binary_label'] == 0).sum()}, Abnormal: {(labels_df['binary_label'] == 1).sum()}")
    return labels_df

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image from the given path.
    Returns a numpy array normalized to [0, 1].
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        # print(f"Loaded image shape: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def get_image_path(image_id, images_dir):
    """
    Given an image filename (e.g., '00000001_000.png'), return the full path in extracted_images.
    """
    path = os.path.join(images_dir, image_id)
    # print(f"Resolved image path: {path}")
    return path

def create_balanced_dataframe(labels_df, random_state=42):
    """
    Create a balanced DataFrame with equal numbers of normal and abnormal samples.
    """
    print("Balancing the dataset...")
    normal_samples = labels_df[labels_df['binary_label'] == 0]
    abnormal_samples = labels_df[labels_df['binary_label'] == 1]
    min_samples = min(len(normal_samples), len(abnormal_samples))
    print(f"Normal samples: {len(normal_samples)}, Abnormal samples: {len(abnormal_samples)}")
    print(f"Balancing to {min_samples} samples per class.")
    balanced_normal = normal_samples.sample(n=min_samples, random_state=random_state)
    balanced_abnormal = abnormal_samples.sample(n=min_samples, random_state=random_state)
    balanced_df = pd.concat([balanced_normal, balanced_abnormal]).reset_index(drop=True)
    print(f"Balanced dataset size: {len(balanced_df)}")
    return balanced_df

def split_data(df, test_size=0.3, val_size=0.5, random_state=42):
    """
    Split the DataFrame into train, validation, and test sets.
    """
    print("Splitting data into train, validation, and test sets...")
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['binary_label'])
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=random_state, stratify=temp_df['binary_label'])
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    return train_df, val_df, test_df
