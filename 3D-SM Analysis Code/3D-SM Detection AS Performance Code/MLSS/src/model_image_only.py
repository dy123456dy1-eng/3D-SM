import torch
import sys
import os
import torch.nn as nn
import torchvision.models as models

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import DEVICE

class ImprovedResNet18(nn.Module):
    """
    Improved ResNet-18 model for feature extraction
    
    Improvements:
    1. Added spatial attention mechanism to enhance key feature weights
    2. Optimized batch normalization configuration for improved training stability
    3. Selectively froze pre-trained parameters for efficient transfer learning
    
    Input: RGB image tensor (batch_size, 3, 224, 224)
    Output: Image feature vector (batch_size, 512)
    """
    def __init__(self, pretrained=True):
        super(ImprovedResNet18, self).__init__()
        # Load pre-trained ResNet-18 model
        # Use new weights parameter instead of deprecated pretrained parameter
        if pretrained:
            # Load ImageNet pre-trained weights
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Remove the final fully connected layer, keep feature extraction part
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add spatial attention mechanism
        # Input: (batch_size, 512, 7, 7) - Output of ResNet18's last convolutional layer
        # Output: (batch_size, 512, 7, 7) - Weighted feature map
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),  # Reduce to single channel attention map
            nn.Sigmoid()  # Normalize to [0,1] range
        )
        
        # Freeze initial convolutional layer parameters (transfer learning strategy)
        # Keep last 10 layers trainable to balance feature reuse and task adaptation
        for param in list(self.features.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        x = self.features(x)
        # Add attention mechanism
        attn = self.attention(x)
        x = x * attn
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten features
        x = x.view(x.size(0), -1)
        return x


class ImageOnlyModel(nn.Module):
    """
    Image-only model for regression prediction
    
    Model architecture:
    1. Use improved ResNet-18 to extract features from 3 images
    2. Fuse all image features and perform regression prediction (output single continuous value)
    """
    def __init__(self, num_classes=1):
        """
        Initialize image-only regression model
        
        Args:
            num_classes (int): Output dimension, set to 1 for regression task
        """
        super(ImageOnlyModel, self).__init__()
        self.num_classes = num_classes
        
        # Image feature extractor (3 identical improved ResNet18)
        self.image_feature_extractor = ImprovedResNet18()
        
        # Feature fusion and regression predictor
        self.regressor = nn.Sequential(
            nn.Linear(512 * 3, 512),  # Features from 3 images
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Regression output (single value)
        )
    
    def forward(self, images):
        """
        Forward propagation
        
        Args:
            images (tensor): Image data, shape is (batch_size, 3, 3, 224, 224)
            
        Returns:
            tensor: Regression prediction result, shape is (batch_size, 1)
        """
        batch_size = images.size(0)
        
        # Extract features from 3 images
        image_features = []
        for i in range(3):  # 3 images
            img = images[:, i, :, :, :]  # (batch_size, 3, 224, 224)
            feature = self.image_feature_extractor(img)  # (batch_size, 512)
            image_features.append(feature)
        
        # Fuse image features
        combined_image_features = torch.cat(image_features, dim=1)  # (batch_size, 512*3)
        
        # Regression prediction
        output = self.regressor(combined_image_features)
        
        return output
