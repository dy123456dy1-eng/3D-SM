import torch
import os

# Get project root directory path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configuration
NUM_CLASSES = 1  # For regression task, output dimension is set to 1
BATCH_SIZE = 32  # Appropriately increase batch size to improve training efficiency
LEARNING_RATE = 0.01  # Initial learning rate
WEIGHT_DECAY = 1e-5  # Reduce weight decay to avoid over-regularization
NUM_EPOCHS = 100  # Adjust training epochs, used with early stopping mechanism

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
