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
    
    # Resize (using BICUBIC for Python 3.6 compatibility)
    img = img.resize(target_size, Image.BICUBIC)
    
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
            
            # Try creating z-only model
            try:
                print("\nCreating z-only model...")
                z_output = None
                for output in model.outputs:
                    if output.name == 'z':
                        z_output = output
                        break
                
                if z_output:
                    # Create model with just z output
                    z_model = C.combine([z_output])
                    
                    # Try different approaches to run it
                    try:
                        print("\nAttempt 1: Using z_model.eval()...")
                        features = model.arguments[0]  # Should be 'features'
                        result = z_model.eval({features: input_data})
                        print(f"Z output shape: {result.shape}")
                        print(f"Z output values: {result.flatten()}")
                        return model
                    except Exception as e1:
                        print(f"Error with z_model.eval(): {e1}")
                        
                        try:
                            print("\nAttempt 2: Extracting computation directly...")
                            # Try to find and run just the 'z' computation node
                            z_node = model.find_by_name('z')
                            if z_node:
                                z_func = C.as_composite(z_node)
                                result = z_func.eval({features: input_data})
                                print(f"Z node output shape: {result.shape}")
                                print(f"Z node output values: {result.flatten()}")
                            else:
                                print("Could not find 'z' node")
                        except Exception as e2:
                            print(f"Error extracting z computation: {e2}")
                
                # Fallback: try providing dummy regr input
                try:
                    print("\nAttempt 3: Using dummy regr input...")
                    features = model.arguments[0]  # 'features'
                    regr = model.arguments[1]      # 'regr'
                    
                    # Create dummy regr input (zeros)
                    dummy_regr = np.zeros((1, 12), dtype=np.float32)
                    
                    # Run full model with both inputs
                    result = model.eval({features: input_data, regr: dummy_regr})
                    
                    # Print all outputs
                    for i, output in enumerate(model.outputs):
                        print(f"{output.name} output shape: {result[i].shape}")
                        print(f"{output.name} output values: {result[i].flatten()}")
                        
                        # Save to file if it's the z output
                        if output.name == 'z':
                            z_values = result[i].flatten()
                            with open('z_output_values.txt', 'w') as f:
                                for j, val in enumerate(z_values):
                                    f.write(f"Vector[{j}] = {val:.7f}\n")
                            print(f"Saved z output values to z_output_values.txt")
                except Exception as e3:
                    print(f"Error with dummy regr input: {e3}")
            except Exception as e:
                print(f"Error creating z-only model: {e}")
        
        return model
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_cntk_model.py <path_to_cntk_model> [image_path]")
        sys.exit(1)
    
    # Model path is required
    model_path = sys.argv[1]
    
    # Image path is optional
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Inspect model
    model = inspect_cntk_model(model_path, image_path) 