import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib
from torchvision import transforms
from PIL import Image

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import DEVICE, MODEL_SAVE_DIR, RESULTS_DIR, NUM_CLASSES
from model_image_only import ImageOnlyModel

class Predictor:
    """
    Image-only model predictor for regression prediction using image data
    
    Core functions:
    - Load trained ResNet18 image-only model
    - Preprocess input image data
    - Execute inference and return readable prediction results
    - Support CSV export
    
    Attributes:
        device: Inference device (GPU preferred)
        model: Loaded image-only model
        transform: Image preprocessing pipeline
    """
    def __init__(self, model_path=None):
        # Initialize model (use regression task, output dimension is 1)
        self.model = ImageOnlyModel(num_classes=1).to(DEVICE)
        # Load model weights
        if model_path is None:
            # Default use best model (optimal model saved during training)
            model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found, please train the model first")
        
        # Load model weights and map to specified device
        # Note: Model state dict was saved directly during training, not the entire checkpoint
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        # Set to evaluation mode (disable dropout and batch normalization training mode)
        self.model.eval()
        
        # Define image preprocessing pipeline (consistent with training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       # Resize to model input size
            transforms.ToTensor(),               # Convert to tensor and normalize to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistical parameters
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_new_data(self, data_dir):
        """
        Load new dataset (image-only data)
        Args: data_dir - Directory containing images and CSV files
        Returns: Processed image data and sample IDs
        """
        # Find CSV file (assume only one CSV file in directory)
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"CSV file not found in {data_dir}")
        csv_path = os.path.join(data_dir, csv_files[0])
        
        # Read CSV data
        data_df = pd.read_csv(csv_path)
        # Check if CSV format is correct (should have 2 columns: group ID, MLSS value)
        if data_df.shape[1] < 2:
            raise ValueError(f"CSV file format error, should have at least 2 columns (group ID, MLSS value), actual {data_df.shape[1]} columns")
        
        samples = []
        sample_ids = []
        
        for idx in range(len(data_df)):
            # Get group ID
            group_id = data_df.iloc[idx, 0]
            
            # Load 3 images (consistent with training)
            images = []
            for img_idx in range(1, 4):
                img_name = f"{group_id}-{img_idx}.png"
                img_path = os.path.join(data_dir, img_name)
                
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file {img_path} not found")
                
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                images.append(image)
            
            # Convert to tensor
            images_tensor = torch.stack(images)
            
            samples.append(images_tensor)
            sample_ids.append(group_id)
        
        return samples, sample_ids

    def predict(self, data_dir, output_csv=False):
        """
        Perform regression prediction on new dataset (image-only)
        Args: 
            data_dir - Directory containing images and CSV files
            output_csv - Whether to save prediction results as CSV file (default False, decided by caller)
        Returns: Prediction results dictionary
        """
        # Load data
        samples, sample_ids = self.load_new_data(data_dir)
        
        predictions = []
        
        with torch.no_grad():
            for images in samples:
                # Add batch dimension and move to device
                images = images.unsqueeze(0).to(DEVICE)
                
                # Get model output directly (regression task doesn't need softmax)
                output = self.model(images)
                # Get predicted value (regression task directly outputs numeric value)
                pred_value = output.item()
                
                predictions.append(pred_value)
        
        # Organize results
        results = pd.DataFrame({
            'group_id': sample_ids,
            'predicted_value': predictions
        })
        
        # Save as CSV file (if needed)
        if output_csv:
            output_path = os.path.join(data_dir, 'predictions.csv')
            results.to_csv(output_path, index=False)
            print(f"Prediction results saved to: {output_path}")
        
        return results


def main():
    """
    Main function: Load model and perform regression prediction
    """
    import argparse
    parser = argparse.ArgumentParser(description='Image-only regression model prediction')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory path, should contain images and CSV files')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model file path')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Result save directory')
    
    # Parse command line arguments, use default values if no arguments provided
    try:
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails, use default arguments
        args = parser.parse_args([])
    
    # Create result directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor (regression task num_classes=1)
    predictor = Predictor(model_path=args.model_path)
    
    # Perform prediction (predict method doesn't save results by default, decided by main function)
    results = predictor.predict(args.data_dir)
    
    # Save results to specified output directory
    output_path = os.path.join(args.output_dir, 'predictions.csv')
    results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Prediction results saved to: {output_path}")
    
    print("Regression prediction completed!")


if __name__ == '__main__':
    main()
