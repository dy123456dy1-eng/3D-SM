import os
from PIL import Image

def split_and_resize_image(image_path, output_dir):
    """
    Split a single image into 3 equal horizontal parts and resize to 224x224
    
    Args:
        image_path (str): Path to the original image
        output_dir (str): Directory to save processed images
    """
    # Get original filename (without extension)
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    
    # Open original image
    with Image.open(image_path) as img:
        # Get image dimensions
        width, height = img.size
        
        # Calculate width for each part
        piece_width = width // 3
        
        # Create 3 filenames (x-1, x-2, x-3)
        filenames = [f"{filename_without_ext}-1.png", f"{filename_without_ext}-2.png", f"{filename_without_ext}-3.png"]
        
        # Split and save 3 parts
        for i in range(3):
            # Calculate crop area
            left = i * piece_width
            upper = 0
            right = (i + 1) * piece_width
            lower = height
            
            # Crop image
            piece = img.crop((left, upper, right, lower))
            
            # Resize to 224x224
            resized_piece = piece.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Save image
            output_path = os.path.join(output_dir, filenames[i])
            resized_piece.save(output_path, "PNG")
            
            print(f"Saved: {filenames[i]}")

def process_all_images():
    """
    Process all images
    """
    # Set input and output directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "images")
    output_dir = os.path.join(current_dir, "processed_images")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort by name to ensure consistent processing order
    image_files.sort()
    
    print(f"Found {len(image_files)} images, starting processing...")
    
    # Process each image
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        split_and_resize_image(image_path, output_dir)
        print(f"Processing completed: {filename}")
    
    print(f"All images processed! Total: {len(image_files)} images, generated {len(image_files) * 3} split images.")
    print(f"Processed images saved to: {output_dir}")

if __name__ == "__main__":
    process_all_images()
