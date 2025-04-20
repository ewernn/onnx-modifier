# test_onnx.py
import numpy as np
import onnxruntime as rt
import argparse
import os

def test_model(model_path):
    print(f"\nTesting model: {model_path}")
    
    # Load model
    print("Loading model...")
    sess = rt.InferenceSession(model_path)
    
    # Get input details
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    # Create random input
    print("\nGenerating test input...")
    test_input = np.random.rand(1,1,1,299,299).astype(np.float32)
    print(f"Input shape: {test_input.shape}")
    print(f"Input sample values: {test_input[0,0,0,0,:5]}")  # Show first 5 values
    
    # Run inference
    print("\nRunning inference...")
    output_name = sess.get_outputs()[0].name
    outputs = sess.run([output_name], {input_name: test_input})
    output = outputs[0]
    
    # Analyze output
    print(f"Output name: {output_name}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample values: {output[0,0,0]}")  # Show first few values
    
    # Check for all zeros
    zero_count = np.sum(output == 0)
    total_elements = output.size
    zero_percentage = (zero_count / total_elements) * 100
    
    print(f"\nAnalysis:")
    print(f"Total elements: {total_elements}")
    print(f"Zero elements: {zero_count}")
    print(f"Percentage zeros: {zero_percentage:.2f}%")
    
    if zero_percentage > 99:
        print("\n⚠️ WARNING: Output appears to be mostly zeros!")
    else:
        print("\n✓ Output contains non-zero values")
    
    # Basic statistics
    print(f"\nOutput statistics:")
    print(f"Min value: {np.min(output)}")
    print(f"Max value: {np.max(output)}")
    print(f"Mean value: {np.mean(output)}")
    print(f"Standard deviation: {np.std(output)}")

def main():
    parser = argparse.ArgumentParser(description='Test ONNX model outputs')
    parser.add_argument('model_path', help='Path to the ONNX model')
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    test_model(args.model_path)

if __name__ == "__main__":
    main()