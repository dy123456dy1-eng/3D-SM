import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from torchvision import transforms
from PIL import Image

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_DIR, CSV_PATH, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR, NUM_CLASSES
from model_image_only import ImageOnlyModel

class Predictor:
    """
    Image-only model predictor for classification prediction using only image data
    
    Core functions:
    - Load trained ResNet34 image classification model
    - Preprocess input image data
    - Execute inference and return classification prediction results
    - Support result visualization and CSV export
    
    Attributes:
        device: Inference device (GPU preferred)
        model: Loaded image classification model
        transform: Image preprocessing pipeline
        label_encoder: Label encoder for converting numeric labels back to original classes
    """
    def __init__(self, model_path=None):
        # Initialize model (use classification task, output dimension is NUM_CLASSES)
        self.model = ImageOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
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
        
        # Load label encoder
        label_encoder_path = os.path.join(RESULTS_DIR, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            print(f"Warning: Label encoder file {label_encoder_path} not found")
            self.label_encoder = None
        
        # Define image preprocessing pipeline (consistent with training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       # Resize to model input size
            transforms.ToTensor(),               # Convert to tensor and normalize to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistical parameters
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_new_data(self, data_dir):
        """
        Load new dataset (for prediction)
        
        Args: 
            data_dir - Directory containing images and CSV files
        
        Returns: 
            samples - Processed image data list
            sample_ids - Sample ID list (group IDs)
            csv_data - Original CSV data (for saving results)
        """
        # Find CSV file (assume only one CSV file in directory)
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"CSV file not found in {data_dir}")
        csv_path = os.path.join(data_dir, csv_files[0])
        
        # Read CSV data (skip first row description, read data from second row)
        # Specify encoding='gbk' to support Chinese
        data_df = pd.read_csv(csv_path, encoding='gbk', skiprows=1, header=None)
        
        # Check if CSV format is correct (should have at least 2 columns: group ID, class label)
        if data_df.shape[1] < 2:
            raise ValueError(f"CSV file format error, should have at least 2 columns (group ID, class label), actual {data_df.shape[1]} columns")
        
        samples = []
        sample_ids = []
        
        for idx in range(len(data_df)):
            # Get group ID (first column)
            group_id = data_df.iloc[idx, 0]
            
            # Load 3 images (consistent with training)
            images = []
            for img_idx in range(1, 4):
                img_name = f"{int(group_id)}-{img_idx}.png"
                img_path = os.path.join(data_dir, img_name)
                
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file {img_path} not found")
                
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                images.append(image)
            
            # Convert to tensor
            images_tensor = torch.stack(images)
            
            samples.append(images_tensor)
            sample_ids.append(int(group_id))
        
        return samples, sample_ids, data_df

    def predict(self, data_dir, output_csv=True, output_dir=None):
        """
        Perform classification prediction on new dataset
        
        Args: 
            data_dir - Directory containing images and CSV files
            output_csv - Whether to save prediction results as CSV file
            output_dir - Result save directory (if None, save to data_dir)
        
        Returns: Prediction results DataFrame
        """
        # Load data
        samples, sample_ids, csv_data = self.load_new_data(data_dir)
        
        predictions = []
        prediction_probs = []
        
        with torch.no_grad():
            for images in samples:
                # Add batch dimension and move to device
                images = images.unsqueeze(0).to(DEVICE)
                
                # Get model output
                output = self.model(images)
                
                # Apply softmax to get probability
                probabilities = torch.softmax(output, dim=1)
                
                # Get predicted class (class with maximum probability)
                pred_class = torch.argmax(probabilities, dim=1).item()
                
                # Get prediction probability
                pred_prob = probabilities[0][pred_class].item()
                
                predictions.append(pred_class)
                prediction_probs.append(pred_prob)
        
        # Organize results
        results = pd.DataFrame({
            'group_id': sample_ids,
            'predicted_class': predictions,
            'prediction_probability': prediction_probs
        })
        
        # If label encoder exists, add original class labels
        if self.label_encoder is not None:
            original_labels = self.label_encoder.inverse_transform(predictions)
            results['predicted_label'] = original_labels
        
        # Determine output directory
        if output_dir is None:
            output_dir = data_dir
        
        # Save as CSV file
        if output_csv:
            output_path = os.path.join(output_dir, 'predictions.csv')
            os.makedirs(output_dir, exist_ok=True)
            results.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Prediction results saved to: {output_path}")
        
        return results

    def visualize_predictions(self, group_ids, predictions, save_path='results/predictions.png'):
        """
        Visualize classification prediction results
        
        Args:
            group_ids (list): Group ID list
            predictions (list): Prediction results list (class labels)
            save_path (str): Save path
        """
        plt.figure(figsize=(12, 6))
        
        # Plot prediction results bar chart
        plt.subplot(1, 2, 1)
        unique_classes = sorted(list(set(predictions)))
        class_counts = [predictions.count(cls) for cls in unique_classes]
        plt.bar(unique_classes, class_counts, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Classes')
        
        # Plot prediction results for each sample
        plt.subplot(1, 2, 2)
        x_pos = range(len(group_ids))
        plt.scatter(x_pos, predictions, s=100, c=predictions, cmap='viridis', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Predicted Class')
        plt.title('Predicted Classes for Each Sample')
        plt.xticks(x_pos, group_ids, rotation=45)
        plt.colorbar(label='Class')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Prediction results plot saved to: {save_path}")


def main():
    """
    Main function: Load model and perform classification prediction
    
    Usage example:
        python predict.py --data_dir "../data" --output_dir "../results"
        or run directly: python predict.py (use default paths)
    
    Args:
        --data_dir: Data directory containing images and CSV files (default: ../data)
        --model_path: Model file path (optional, default: models/saved/best_model.pth)
        --output_dir: Result save directory (optional, default: results)
    """
    import argparse
    parser = argparse.ArgumentParser(description='Image-only classification model prediction')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory path, should contain images and CSV files')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model file path')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Result save directory')
    
    args = parser.parse_args()
    
    # If data_dir not specified, use default path (data folder in parent directory of script file)
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, '..', 'data')
        args.data_dir = os.path.normpath(args.data_dir)
    
    # Create result directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor (classification task num_classes=NUM_CLASSES)
    predictor = Predictor(model_path=args.model_path)
    
    # Perform prediction, results saved to specified directory
    results = predictor.predict(args.data_dir, output_csv=True, output_dir=args.output_dir)
    
    print("\nClassification prediction completed!")
    print(f"Prediction results saved to: {os.path.join(args.output_dir, 'predictions.csv')}")


if __name__ == '__main__':
    main()
