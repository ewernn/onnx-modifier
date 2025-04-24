import os
import numpy as np
import onnxruntime as rt
import onnx
import logging
import sys
import warnings
from PIL import Image
from onnx_modifier.onnx_modifier import onnxModifier

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to redirect ONNX Runtime warnings to a file
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onnx_runtime.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress ONNX Runtime warnings
onnxruntime_logger = logging.getLogger('onnxruntime')
onnxruntime_logger.setLevel(logging.ERROR)

# Suppress ONNX warnings
onnx_logger = logging.getLogger('onnx')
onnx_logger.setLevel(logging.ERROR)

def inspect_model(model_path):
    """Inspect ONNX model structure and print information."""
    try:
        model = onnx.load(model_path)
        print("\n=== Model Inspection ===")
        
        # Show inputs
        print("\nInputs:")
        for input in model.graph.input:
            print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        
        # Show outputs
        print("\nOutputs:")
        for output in model.graph.output:
            print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
        
        # Find auxiliary branch nodes
        print("\nAuxiliary Branch Nodes:")
        aux_nodes = []
        for node in model.graph.node:
            # Look for nodes connected to regr or rmse_Output
            if any('regr' in inp for inp in node.input) or any('rmse' in out for out in node.output):
                aux_nodes.append(node)
                print(f"  {node.name} ({node.op_type})")
                print(f"    Inputs: {node.input}")
                print(f"    Outputs: {node.output}")
        
        # Count initializers that are also inputs
        initializer_names = {init.name for init in model.graph.initializer}
        input_names = {input.name for input in model.graph.input}
        problematic_inputs = initializer_names.intersection(input_names)
        
        if problematic_inputs:
            print(f"\nWARNING: Found {len(problematic_inputs)} initializers that are also inputs")
        
        return model
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None

def fix_pooling_pads(model_path):
    """Fix pooling layer padding issues in the ONNX model."""
    try:
        # Create ONNX modifier instance
        modifier = onnxModifier.from_name_protobuf_stream(
            os.path.basename(model_path),
            open(model_path, 'rb')
        )
        
        # Fix pooling pads
        was_modified = modifier.fix_pooling_pads()
        if was_modified:
            print("Fixed pooling layer paddings")
            
            # Save fixed model using the correct method
            fixed_path = model_path.replace('.onnx', '_fixed.onnx')
            with open(fixed_path, 'wb') as f:
                f.write(modifier.model_proto.SerializeToString())
            print(f"Saved fixed model to: {fixed_path}")
            return fixed_path
        
        return model_path
    except Exception as e:
        print(f"Error fixing pooling pads: {e}")
        return model_path

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

def load_expected_output(output_path):
    """Load the expected output from file."""
    with open(output_path, 'r') as f:
        expected_output = np.array([float(x.strip()) for x in f.readlines()], dtype=np.float32)
    return expected_output

def test_model(model_path, image_path, expected_output_path):
    """Test ONNX model with specific image and compare with expected output."""
    try:
        # First inspect the model
        print("\n=== Model Inspection ===")
        model = inspect_model(model_path)
        if model is None:
            return False
        
        # # Try to fix pooling pads
        # print("\n=== Fixing Pooling Pads ===")
        # fixed_model_path = fix_pooling_pads(model_path)
        fixed_model_path = model_path
        
        # Load and preprocess image
        input_data = load_and_preprocess_image(image_path)
        print(f"\nInput image shape: {input_data.shape}")
        
        # Create inference session with warning suppression
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3  # Suppress all warnings
        sess = rt.InferenceSession(fixed_model_path, sess_options)
        
        # Get input details
        input_name = sess.get_inputs()[0].name
        print(f"Model input name: {input_name}")
        
        # Run inference
        outputs = sess.run(None, {input_name: input_data})
        
        # Load expected output
        expected_output = load_expected_output(expected_output_path)
        
        # Compare outputs
        for i, output in enumerate(outputs):
            print(f"\nOutput {i} shape: {output.shape}")
            print(f"Output {i} values: {output.flatten()}")
            print(f"Expected output: {expected_output}")
            
            # Calculate difference
            diff = np.abs(output.flatten() - expected_output)
            print(f"Maximum difference: {np.max(diff)}")
            print(f"Mean difference: {np.mean(diff)}")
            
            if np.allclose(output.flatten(), expected_output, rtol=1e-3, atol=1e-3):
                print("Output matches expected values!")
            else:
                print("WARNING: Output does not match expected values!")
        
        return True
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    # Model and data paths
    model_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/fixed_models/1-CanShoulderConds_fixed.onnx"
    model_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/fixed_models/squeezed_1-CanShoulderConds_fixed.onnx"
    model_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/fixed_models/351pm-squeezed_1-CanShoulderConds_fixed.onnx"
    image_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/Sample_Shoulder_Conds.jpg"
    expected_output_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/Shoulder_Conds_Output.txt"
    
    # Run test
    test_model(model_path, image_path, expected_output_path) 
