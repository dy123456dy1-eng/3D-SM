import torch
import os

# Get project root directory path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configuration
NUM_CLASSES = 3  # For classification task, output dimension is set to 3
BATCH_SIZE = 16  # Appropriately reduce batch size to improve model generalization
LEARNING_RATE = 0.001  # Reduce learning rate to avoid skipping optimal solutions
WEIGHT_DECAY = 1e-4  # Increase weight decay to enhance regularization
NUM_EPOCHS = 1000  # Moderate training epochs to balance training time and effectiveness

# Data configuration
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data')  # Image data directory
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'data.csv')  # CSV data file path

# Model save configuration
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create necessary directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
