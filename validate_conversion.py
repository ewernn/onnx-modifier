import os
import numpy as np
import onnx
import onnxruntime as rt
import cntk as C
from onnx_modifier.onnx_modifier import onnxModifier

def load_cntk_model(model_path):
    """Load and validate CNTK model."""
    try:
        model = C.load_model(model_path)
        print(f"Successfully loaded CNTK model: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading CNTK model: {e}")
        return None

def test_onnx_model(model_path, input_shape=(1, 3, 224, 224)):
    """Test ONNX model with random input and check for zero outputs."""
    try:
        # Load model
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        # Create inference session
        sess = rt.InferenceSession(model_path)
        
        # Get input details
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"Model input shape: {input_shape}")
        
        # Create test input
        np.random.seed(42)  # For reproducibility
        x = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = sess.run(None, {input_name: x})
        
        # Check outputs
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i} min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
            if np.allclose(output, 0):
                print(f"WARNING: Output {i} contains all zeros!")
            else:
                print(f"Output {i} contains non-zero values")
        
        return True
    except Exception as e:
        print(f"Error testing ONNX model: {e}")
        return False

def validate_conversion_steps(model_path):
    """Validate model at each stage of conversion."""
    print("\n=== Starting Validation Process ===\n")
    
    # 1. Test original CNTK model
    print("\n1. Testing original CNTK model...")
    cntk_model = load_cntk_model(model_path)
    if cntk_model:
        # Create test input
        np.random.seed(42)
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        try:
            output = cntk_model.eval({cntk_model.arguments[0]: x})
            print(f"CNTK model output shape: {output.shape}")
            print(f"CNTK model output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
        except Exception as e:
            print(f"Error evaluating CNTK model: {e}")
    
    # 2. Test after initial ONNX conversion
    onnx_path = model_path.replace('.dnn', '.onnx')
    print(f"\n2. Testing initial ONNX conversion: {onnx_path}")
    if os.path.exists(onnx_path):
        test_onnx_model(onnx_path)
    else:
        print(f"ONNX model not found: {onnx_path}")
    
    # 3. Test after squeeze/unsqueeze addition
    squeezed_path = onnx_path.replace('.onnx', '_squeezed.onnx')
    print(f"\n3. Testing squeezed version: {squeezed_path}")
    if os.path.exists(squeezed_path):
        test_onnx_model(squeezed_path)
    else:
        print(f"Squeezed model not found: {squeezed_path}")
    
    # 4. Test after ONNX-modifier fixes
    fixed_path = squeezed_path.replace('.onnx', '_fixed.onnx')
    print(f"\n4. Testing fixed version: {fixed_path}")
    if os.path.exists(fixed_path):
        test_onnx_model(fixed_path)
    else:
        print(f"Fixed model not found: {fixed_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate_conversion.py <path_to_cntk_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    validate_conversion_steps(model_path) 