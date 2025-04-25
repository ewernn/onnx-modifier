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
        
        # Build a map of node outputs to their nodes
        output_to_node = {}
        for node in model.graph.node:
            for output in node.output:
                output_to_node[output] = node
        
        # Build a map of node inputs to their nodes
        input_to_node = {}
        for node in model.graph.node:
            for input in node.input:
                if input not in input_to_node:
                    input_to_node[input] = []
                input_to_node[input].append(node)
        
        # Show all outputs and their connections
        print("\nAll Outputs and Their Connections:")
        for output in model.graph.output:
            print(f"\n  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
            
            # Find nodes that produce this output
            if output.name in output_to_node:
                node = output_to_node[output.name]
                print(f"    Produced by: {node.name} ({node.op_type})")
                print(f"    Inputs: {node.input}")
            
            # Find nodes that take this output as input
            if output.name in input_to_node:
                print("    Used by:")
                for node in input_to_node[output.name]:
                    print(f"      {node.name} ({node.op_type})")
        
        # Show nodes near the outputs
        print("\nNodes Near Outputs:")
        for node in model.graph.node:
            # Check if this node's outputs are used by output nodes
            is_near_output = False
            for output in node.output:
                if output in input_to_node:
                    for dependent_node in input_to_node[output]:
                        if dependent_node.output and any(out in [o.name for o in model.graph.output] for out in dependent_node.output):
                            is_near_output = True
                            break
            
            if is_near_output:
                print(f"\n  {node.name} ({node.op_type})")
                print(f"    Inputs: {node.input}")
                print(f"    Outputs: {node.output}")
                # Show attributes for relevant nodes
                if node.op_type in ['MaxPool', 'AveragePool', 'Conv', 'BatchNormalization']:
                    print("    Attributes:")
                    for attr in node.attribute:
                        if attr.type == 7:  # INTS
                            print(f"      {attr.name}: {list(attr.ints)}")
        
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

def load_and_preprocess_image(image_path, target_size=(299, 299), color_mode='L', visualize=True, normalization='simple'):
    """Load image and preprocess with different options.
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        color_mode: 'L' for grayscale, 'RGB' for color
        visualize: Whether to display the image
        normalization: 'simple' for [0,1], 'imagenet' for ImageNet normalization
    """
    # Load image
    img = Image.open(image_path)
    
    # Visualize original image if requested
    if visualize:
        try:
            print(f"\n=== Input Image Visualization ===")
            print(f"Original image shape: {img.size}, mode: {img.mode}")
            img.show()  # Display image
        except Exception as e:
            print(f"Error visualizing image: {e}")
    
    # Convert to desired color mode
    img = img.convert(color_mode)
    if visualize and color_mode != img.mode:
        try:
            print(f"Converted to {color_mode} mode")
            img.show()  # Show converted image
        except Exception as e:
            print(f"Error visualizing converted image: {e}")
    
    # Resize
    img = img.resize(target_size, Image.BICUBIC)
    if visualize:
        try:
            print(f"Resized to {target_size}")
            img.show()  # Show resized image
        except Exception as e:
            print(f"Error visualizing resized image: {e}")
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    if normalization == 'simple':
        # Simple [0,1] normalization
        img_array = img_array / 255.0
    elif normalization == 'imagenet':
        # ImageNet normalization
        if color_mode == 'RGB':
            # ImageNet mean and std for RGB
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
            img_array = img_array / 255.0
            img_array = (img_array - mean) / std
        else:
            # For grayscale, use average of RGB means/stds
            mean = 0.449
            std = 0.226
            img_array = img_array / 255.0
            img_array = (img_array - mean) / std
    elif normalization == 'centered':
        # Center to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
    
    # Add dimensions based on color mode
    if color_mode == 'L':
        # For grayscale: Add batch, channel, and extra dimensions
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add extra dimension for 5D
    else:
        # For RGB: Add batch and extra dimension, channel is already there
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.transpose(img_array, (0, 3, 1, 2))  # NHWC to NCHW
        img_array = np.expand_dims(img_array, axis=0)  # Add extra dimension for 5D
    
    print(f"Final preprocessed shape: {img_array.shape}")
    print(f"Value range: [{img_array.min():.4f}, {img_array.max():.4f}], mean: {img_array.mean():.4f}")
    return img_array

def load_expected_output(output_path):
    """Load the expected output from file."""
    with open(output_path, 'r') as f:
        lines = f.readlines()
        # Parse CNTK output format: "Vector[0] = 0.0000"
        values = []
        for line in lines:
            if '=' in line:
                value = float(line.split('=')[1].strip())
                values.append(value)
        return np.array(values, dtype=np.float32)

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

def inspect_conv_node(model_path, node_name="model.model.conv_1._.x.c"):
    """Inspect a specific Conv node in the model."""
    try:
        model = onnx.load(model_path)
        print(f"\n=== Inspecting Conv Node: {node_name} ===")
        
        # Find the specific node
        target_node = None
        for node in model.graph.node:
            if node.name == node_name:
                target_node = node
                break
        
        if target_node:
            print(f"\nNode Details:")
            print(f"  Name: {target_node.name}")
            print(f"  Op Type: {target_node.op_type}")
            print(f"  Inputs: {target_node.input}")
            print(f"  Outputs: {target_node.output}")
            print("\n  Attributes:")
            for attr in target_node.attribute:
                if attr.type == 7:  # INTS
                    print(f"    {attr.name}: {list(attr.ints)}")
                elif attr.type == 1:  # FLOAT
                    print(f"    {attr.name}: {attr.f}")
                elif attr.type == 2:  # INT
                    print(f"    {attr.name}: {attr.i}")
                elif attr.type == 3:  # STRING
                    print(f"    {attr.name}: {attr.s}")
        else:
            print(f"Node {node_name} not found in model")
        
        return target_node
    except Exception as e:
        print(f"Error inspecting Conv node: {e}")
        return None

def test_model_with_variants(model_path, image_path, expected_output_path):
    """Test the model with different preprocessing variants."""
    # Save original input shape for the first test
    original_shape = None
    best_result = None
    best_difference = float('inf')
    
    # Define preprocessing variants to try
    variants = [
        {"name": "Grayscale, Simple Normalization", "color_mode": "L", "normalization": "simple"},
        {"name": "RGB, Simple Normalization", "color_mode": "RGB", "normalization": "simple"},
        {"name": "Grayscale, ImageNet Normalization", "color_mode": "L", "normalization": "imagenet"},
        {"name": "RGB, ImageNet Normalization", "color_mode": "RGB", "normalization": "imagenet"},
        {"name": "Grayscale, Centered Normalization", "color_mode": "L", "normalization": "centered"},
        {"name": "RGB, Centered Normalization", "color_mode": "RGB", "normalization": "centered"},
    ]
    
    # Only visualize the first variant
    visualize_first = True
    
    # Initialize model only once
    try:
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3  # Suppress all warnings
        sess = rt.InferenceSession(model_path, sess_options)
        input_name = sess.get_inputs()[0].name
        print(f"Model input name: {input_name}")
        
        # Load expected output
        expected_output = load_expected_output(expected_output_path)
        
        # Try each variant
        for i, variant in enumerate(variants):
            print(f"\n\n{'='*20} Testing Variant {i+1}: {variant['name']} {'='*20}")
            
            # Preprocess with this variant
            visualize = visualize_first and i == 0
            input_data = load_and_preprocess_image(
                image_path, 
                color_mode=variant["color_mode"], 
                normalization=variant["normalization"],
                visualize=visualize
            )
            
            # Save shape of first variant for comparison
            if i == 0:
                original_shape = input_data.shape
            elif input_data.shape != original_shape:
                print(f"WARNING: Shape mismatch with original input: {input_data.shape} vs {original_shape}")
                # Continue to next variant if shapes don't match
                continue
            
            # Run inference
            try:
                outputs = sess.run(None, {input_name: input_data})
                
                # Compare outputs
                for j, output in enumerate(outputs):
                    print(f"\nOutput {j} shape: {output.shape}")
                    print(f"Output {j} values: {output.flatten()}")
                    print(f"Expected output: {expected_output}")
                    
                    # Calculate difference
                    diff = np.abs(output.flatten() - expected_output)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"Maximum difference: {max_diff}")
                    print(f"Mean difference: {mean_diff}")
                    
                    # Track best result
                    if mean_diff < best_difference:
                        best_difference = mean_diff
                        best_result = {
                            "variant": variant["name"],
                            "output": output.flatten(),
                            "max_diff": max_diff,
                            "mean_diff": mean_diff
                        }
                    
                    if np.allclose(output.flatten(), expected_output, rtol=1e-3, atol=1e-3):
                        print("SUCCESS: Output matches expected values!")
                    else:
                        print("WARNING: Output does not match expected values!")
            except Exception as e:
                print(f"Error running inference with variant {i+1}: {e}")
        
        # Summarize best result
        if best_result:
            print(f"\n\n{'='*20} Best Result {'='*20}")
            print(f"Best variant: {best_result['variant']}")
            print(f"Output: {best_result['output']}")
            print(f"Maximum difference: {best_result['max_diff']}")
            print(f"Mean difference: {best_result['mean_diff']}")
            
            if best_result['mean_diff'] < 0.1:
                print("CONCLUSION: Found a good match!")
            else:
                print("CONCLUSION: No variant provided a good match.")
        
        return True
    except Exception as e:
        print(f"Error in preprocessing variants test: {e}")
        return False

if __name__ == "__main__":
    model_path = "/Users/ewern/Downloads/modified_modified_12pm-squeezed_CanShoulderConds_fixed.onnx"
    image_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/Sample_Shoulder_Conds.jpg"
    expected_output_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/Shoulder_Conds_Output.txt"
    
    # First inspect the problematic Conv node
    inspect_conv_node(model_path)
    
    # Test with different preprocessing variants
    test_model_with_variants(model_path, image_path, expected_output_path)
    
    # Run original test if needed
    # test_model(model_path, image_path, expected_output_path) 
