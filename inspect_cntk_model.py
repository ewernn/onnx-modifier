import os
import numpy as np
import cntk as C
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    """Load image, convert to grayscale, and resize."""
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def inspect_cntk_model(model_path, image_path=None):
    """Inspect CNTK model structure and optionally test it with an image."""
    print(f"\nLoading CNTK model from: {model_path}")
    try:
        # Load model
        model = C.load_model(model_path)
        print("\n=== Model Structure ===")
        
        # Get model arguments (inputs)
        print("\nInputs:")
        for arg in model.arguments:
            print(f"  Name: {arg.name}")
            print(f"  Shape: {arg.shape}")
        
        # Get model outputs
        print("\nOutputs:")
        for output in model.outputs:
            print(f"  Name: {output.name}")
            print(f"  Shape: {output.shape}")
        
        # If image path is provided, test the model
        if image_path:
            print("\n=== Testing Model with Image ===")
            # Load and preprocess image
            input_data = load_and_preprocess_image(image_path)
            print(f"Input image shape: {input_data.shape}")
            
            # Run inference
            output = model.eval({model.arguments[0]: input_data})
            print(f"\nModel output shape: {output.shape}")
            print(f"Model output values: {output.flatten()}")
        
        return model
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None

if __name__ == "__main__":
    # Model and data paths
    model_path = "/home/ewern/onnx-modifier/apr24/1-CanShoulderConds"
    image_path = "/home/ewern/onnx-modifier/apr24/Sample_Shoulder_Conds.jpg"
    
    # Inspect model
    model = inspect_cntk_model(model_path, image_path) 