import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    Image-only dataset loading class for classification tasks
    
    Data format requirements:
    - CSV file must contain at least 4 columns: [Group ID, Numeric Metric 1, Numeric Metric 2, Target Value]
    - Target value should be class labels (e.g., 0, 1, 2 representing 3 different classes)
    - Image file naming format: {group_id}-{index}.png (index 1-3, corresponding to 3 images)
    
    Processing flow:
    1. Read CSV file and parse target values
    2. Use LabelEncoder to encode target values (ensure classes are consecutively numbered starting from 0)
    3. Match and load 3 images by group ID and apply preprocessing
    
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
        
        # For classification task, extract the last column as class labels and use LabelEncoder for encoding
        raw_targets = self.data_df.iloc[:, -1].values
        self.label_encoder = LabelEncoder()
        self.targets = self.label_encoder.fit_transform(raw_targets).astype(np.int64)
        
        # Record number of classes
        self.num_classes = len(self.label_encoder.classes_)
        
        # Save label encoder for later use
        joblib.dump(self.label_encoder, os.path.join(RESULTS_DIR, 'label_encoder.pkl'))

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get group ID
        group_id = self.data_df.iloc[idx, 0]
        # Get classification target value
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

        # Convert target value to tensor (classification task uses long integer)
        target = torch.tensor(target, dtype=torch.long)

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
    # Create complete dataset instance
    full_dataset = ImageOnlyDataset()
    
    # Get number of classes
    num_classes = full_dataset.num_classes
    
    # Split training and testing sets (8:2 ratio)
    full_size = len(full_dataset)
    train_size = int(0.8 * full_size)
    test_size = full_size - train_size
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Create weighted sampler for training set to handle class imbalance
    # Calculate weight for each sample
    train_targets = []
    for i in range(len(train_dataset)):
        sample_idx = train_dataset.indices[i]  # Get original dataset index
        target = full_dataset[sample_idx]['target'].item()
        train_targets.append(target)
    
    # Calculate number of samples for each class
    class_counts = torch.bincount(torch.tensor(train_targets))
    # Calculate class weights (inversely proportional to sample count)
    class_weights = 1.0 / class_counts.float()
    # Assign weight to each sample
    sample_weights = torch.tensor([class_weights[t] for t in train_targets])
    
    # Create weighted random sampler
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes
