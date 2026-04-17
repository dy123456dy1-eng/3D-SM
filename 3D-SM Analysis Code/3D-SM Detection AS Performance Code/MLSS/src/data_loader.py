import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import IMAGE_DIR, CSV_PATH, BATCH_SIZE, RESULTS_DIR

class ImageOnlyDataset(Dataset):
    """
    Image-only dataset loading class for regression tasks
    
    Data format requirements:
    - CSV file must contain at least 4 columns: [Group ID, Numeric Metric 1, Numeric Metric 2, Target Value]
    - Image file naming format: {group_id}-{index}.png (index 1-3, corresponding to 3 images)
    
    Processing flow:
    1. Read CSV file and parse target values
    2. Match and load 3 images by group ID and apply preprocessing
    
    Args:
        transform: Image preprocessing transformation chain, default uses ImageNet normalization
    
    Output data structure:
        {'images': tensor [3, 3, 224, 224], 'target': tensor [], 'num_classes': int}
    """
    def __init__(self, transform=None):
        # Read CSV file, format: [Group ID, Metric 1, Metric 2, ..., Label]
        # Specify encoding as GBK to solve Chinese decoding issues
        self.data_df = pd.read_csv(CSV_PATH, encoding='gbk')
        
        # Define image preprocessing pipeline
        # Default uses ResNet standard preprocessing: 224x224 resize + normalization
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to ResNet input size
            transforms.ToTensor(),          # Convert to tensor and normalize to [0,1]
            # ImageNet statistical parameters normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For regression task, directly use the last column as target value (no label encoder needed)
        # Extract the last column of CSV as regression target value
        self.targets = self.data_df.iloc[:, -1].values.astype(np.float32)
        
        # Record output dimension (1 for regression task)
        self.num_classes = 1

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get group ID
        group_id = self.data_df.iloc[idx, 0]
        # Get regression target value
        target = self.targets[idx]

        # Load 3 images (assume each group has 3 images, indexed 1-3)
        images = []
        for img_idx in range(1, 4):  # Modified to load 3 images
            img_name = f"{group_id}-{img_idx}.png"
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            try:
                # Check if file exists
                if not os.path.exists(img_path):
                    raise ValueError(f"Image file {img_path} not found, please check file naming")
                image = Image.open(img_path).convert('RGB')  # Convert to RGB format
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                raise ValueError(f"Failed to load image file {img_path}: {str(e)}")

        # Convert target value to tensor (regression task uses float type)
        target = torch.tensor(target, dtype=torch.float32)

        return {
            'images': torch.stack(images),  # Shape: [3, 3, 224, 224]
            'target': target,               # Shape: []
            'num_classes': self.num_classes # Output dimension
        }


def get_data_loaders():
    """
    Get data loaders for training and testing sets
    Returns: train_loader, test_loader, num_classes
    """
    # Create training and testing set instances
    train_dataset = ImageOnlyDataset()
    test_dataset = ImageOnlyDataset()
    
    # Get number of classes
    num_classes = train_dataset.num_classes
    
    # Split training and testing sets (8:2 ratio)
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes
