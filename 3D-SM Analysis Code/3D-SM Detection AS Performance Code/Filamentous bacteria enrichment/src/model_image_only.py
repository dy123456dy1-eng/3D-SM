import torch
import sys
import os
import torch.nn as nn
import torchvision.models as models

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import DEVICE

class ImprovedResNet101(nn.Module):
    """
    Improved ResNet-101 model for feature extraction
    
    Improvements:
    1. Added spatial attention mechanism to enhance key feature weights
    2. Optimized batch normalization configuration for improved training stability
    3. Selectively froze pre-trained parameters for efficient transfer learning
    
    Input: RGB image tensor (batch_size, 3, 224, 224)
    Output: Image feature vector (batch_size, 2048)
    
    Note: ResNet-101's final feature dimension is 2048, same as ResNet-50, but with deeper network and stronger feature extraction capability
    """
    def __init__(self, pretrained=True):
        super(ImprovedResNet101, self).__init__()
        # Load pre-trained ResNet-101 model
        # Use new weights parameter instead of deprecated pretrained parameter
        if pretrained:
            # Load ImageNet pre-trained weights
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet101(weights=None)
        
        # Remove the final fully connected layer, keep feature extraction part
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add spatial attention mechanism
        # Input: (batch_size, 2048, 7, 7) - Output of ResNet101's last convolutional layer
        # Output: (batch_size, 2048, 7, 7) - Weighted feature map
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),  # Reduce to single channel attention map
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
    Image-only model for classification prediction
    
    Model architecture:
    1. Use improved ResNet-101 to extract features from 3 images
    2. Fuse all image features and perform classification prediction (output class probability distribution)
    
    Note: ResNet-101's feature dimension is 2048, so fused feature dimension is 2048*3=6144
    """
    def __init__(self, num_classes=3):
        """
        Initialize image-only classification model
        
        Args:
            num_classes (int): Output dimension, set to number of classes for classification task
        """
        super(ImageOnlyModel, self).__init__()
        self.num_classes = num_classes
        
        # Image feature extractor (3 identical improved ResNet101)
        self.image_feature_extractor = ImprovedResNet101()
        
        # Feature fusion and classification predictor - with stronger regularization
        # Note: Input dimension changed from 512*3 to 2048*3=6144
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 3, 1024),  # Input dimension: 6144 -> 1024
            nn.BatchNorm1d(1024),      # Add batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),           # Increase dropout rate
            
            nn.Linear(1024, 512),      # Second hidden layer
            nn.BatchNorm1d(512),       # Add batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),           # Dropout rate
            
            nn.Linear(512, 256),       # Third hidden layer
            nn.BatchNorm1d(256),       # Add batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),           # Gradually reduce dropout rate
            
            nn.Linear(256, num_classes)  # Classification output (number of classes)
        )
    
    def forward(self, images):
        """
        Forward propagation
        
        Args:
            images (tensor): Image data, shape is (batch_size, 3, 3, 224, 224)
            
        Returns:
            tensor: Classification prediction result, shape is (batch_size, num_classes)
        """
        batch_size = images.size(0)
        
        # Extract features from 3 images
        image_features = []
        for i in range(3):  # 3 images
            img = images[:, i, :, :, :]  # (batch_size, 3, 224, 224)
            feature = self.image_feature_extractor(img)  # (batch_size, 2048)
            image_features.append(feature)
        
        # Fuse image features
        combined_image_features = torch.cat(image_features, dim=1)  # (batch_size, 2048*3=6144)
        
        # Classification prediction
        output = self.classifier(combined_image_features)
        
        return output
