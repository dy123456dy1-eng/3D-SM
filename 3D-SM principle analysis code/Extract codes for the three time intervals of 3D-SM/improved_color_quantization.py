import os
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def calculate_red_bias_improved(image_path):
    """
    Improved version of red bias calculation function
    Uses a more stable method to calculate the red bias of an image
    """
    try:
        # Open image and convert to RGB mode
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Separate RGB three channels
        r_channel = img_array[:, :, 0].astype(np.float64)
        g_channel = img_array[:, :, 1].astype(np.float64)
        b_channel = img_array[:, :, 2].astype(np.float64)
        
        # Calculate total intensity for each pixel
        total_intensity = r_channel + g_channel + b_channel
        
        # Avoid division by zero - use a very small value instead of 0
        epsilon = 1e-8
        total_intensity = np.where(total_intensity == 0, epsilon, total_intensity)
        
        # Method 1: Red relative intensity ratio (Red/(Green+Blue+small constant))
        divisor = g_channel + b_channel + epsilon
        red_bias1 = (r_channel / divisor) * 2  # Multiply by 2 to balance the ratio
        
        # Method 2: Red proportion (Red/Total intensity)
        red_ratio = r_channel / total_intensity
        
        # Method 3: Red dominance (Red - max(Green, Blue))
        red_excess = r_channel - np.maximum(g_channel, b_channel)
        
        # Method 4: Red saturation (Calculate color saturation, weighted by red saturation)
        max_channel = np.maximum(np.maximum(r_channel, g_channel), b_channel)
        min_channel = np.minimum(np.minimum(r_channel, g_channel), b_channel)
        saturation = (max_channel - min_channel) / (max_channel + epsilon)
        red_saturation = saturation * (r_channel / total_intensity)  # Saturation weight of red channel
        
        # Calculate average values of each metric
        avg_red_bias1 = np.mean(red_bias1)
        avg_red_ratio = np.mean(red_ratio)
        avg_red_excess = np.mean(red_excess)
        avg_red_saturation = np.mean(red_saturation)
        
        # Calculate final quantized value using weighted average
        # Adjust weights to ensure even slight red bias can be captured
        quantized_value = (
            avg_red_bias1 * 0.3 +      # Relative intensity ratio
            avg_red_ratio * 0.25 +     # Red proportion
            avg_red_excess * 0.25 +    # Red dominance
            avg_red_saturation * 0.2   # Red saturation
        )
        
        # Ensure result is non-negative
        quantized_value = max(0, quantized_value)
        
        return quantized_value
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0.0

def calculate_red_bias_log_based(image_path):
    """
    Calculate red bias using logarithmic method (more stable method)
    """
    try:
        # Open image and convert to RGB mode
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Separate RGB three channels
        r_channel = img_array[:, :, 0].astype(np.float64)
        g_channel = img_array[:, :, 1].astype(np.float64)
        b_channel = img_array[:, :, 2].astype(np.float64)
        
        # Use logarithmic method for more stable processing of different red intensities
        # Avoid division by zero
        epsilon = 1e-8
        
        # Calculate the ratio of red to other colors (logarithmic scale)
        red_vs_green = np.log(r_channel + 1) - np.log(g_channel + 1)
        red_vs_blue = np.log(r_channel + 1) - np.log(b_channel + 1)
        
        # Calculate average red advantage
        avg_red_advantage = np.mean(red_vs_green + red_vs_blue)
        
        # Normalize result to non-negative range
        # Use sigmoid-like function to ensure output is in reasonable range
        if avg_red_advantage > 0:
            quantized_value = avg_red_advantage
        else:
            quantized_value = 0.0
            
        return quantized_value
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0.0

def calculate_red_bias_histogram(image_path):
    """
    Calculate red bias using histogram method (more robust method)
    """
    try:
        # Open image and convert to RGB mode
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Separate RGB three channels
        r_channel = img_array[:, :, 0].astype(np.float64)
        g_channel = img_array[:, :, 1].astype(np.float64)
        b_channel = img_array[:, :, 2].astype(np.float64)
        
        # Calculate average values for each channel
        avg_r = np.mean(r_channel)
        avg_g = np.mean(g_channel)
        avg_b = np.mean(b_channel)
        
        # Calculate red bias (more robust method)
        # Avoid division by zero
        epsilon = 1e-8
        
        # Method 1: Red difference relative to other colors
        red_vs_others = avg_r - ((avg_g + avg_b) / 2)
        
        # Method 2: Red proportion
        total_avg = avg_r + avg_g + avg_b
        red_proportion = avg_r / (total_avg + epsilon)
        
        # Method 3: Red dominance
        red_dominance = avg_r - max(avg_g, avg_b) if max(avg_g, avg_b) > 0 else avg_r
        
        # Comprehensive calculation
        # Ensure non-zero output even for slight red bias
        quantized_value = max(0, red_vs_others * 0.4 + red_proportion * 255 * 0.3 + red_dominance * 0.3)
        
        return quantized_value
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0.0

def main():
    """
    Main function: Traverse image folder, calculate chromaticity quantization values, and generate Excel table
    """
    # Specify image folder path (relative path: relative to current script directory)
    image_folder = 'Case image'
    
    # Get all image files in the folder
    image_files = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            image_files.append(file)
    
    print(f"Found {len(image_files)} image files")
    
    # Store results
    results = []
    
    # Traverse each image, calculate chromaticity quantization values
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        
        # Use improved method to calculate red bias
        quantized_value = calculate_red_bias_histogram(image_path)
        
        results.append({
            'Image Name': image_file,
            'Chromaticity Quantization Value': quantized_value
        })
        
        # Display progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by chromaticity quantization value in descending order
    df = df.sort_values(by='Chromaticity Quantization Value', ascending=False)
    
    # Save to Excel file (relative path: output file saved in current directory)
    output_file = 'improved_color_quantization_results.xlsx'
    df.to_excel(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total processed {len(results)} images")
    
    # Display some statistical information
    print(f"\nStatistical Information:")
    print(f"Quantization value range: {df['Chromaticity Quantization Value'].min():.6f} - {df['Chromaticity Quantization Value'].max():.6f}")
    print(f"Mean: {df['Chromaticity Quantization Value'].mean():.6f}")
    print(f"Median: {df['Chromaticity Quantization Value'].median():.6f}")
    
    # Count zero values
    zero_count = len(df[df['Chromaticity Quantization Value'] == 0])
    print(f"Number of images with quantization value of 0: {zero_count}")
    print(f"Zero value ratio: {zero_count/len(df)*100:.2f}%")
    
    # Display top 10 and bottom 10 results
    print(f"\nTop 10 reddest images:")
    for idx, row in df.head(10).iterrows():
        print(f"  {row['Image Name']}: {row['Chromaticity Quantization Value']:.6f}")
    
    print(f"\nBottom 10 least red images:")
    for idx, row in df.tail(10).iterrows():
        print(f"  {row['Image Name']}: {row['Chromaticity Quantization Value']:.6f}")

if __name__ == "__main__":
    main()
