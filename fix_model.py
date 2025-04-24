import os
from onnx_modifier.onnx_modifier import onnxModifier

def fix_model(input_model_path, output_dir='./fixed_models'):
    """Fix ONNX model by removing auxiliary branch and fixing pooling pads."""
    print(f"Loading model from: {input_model_path}")
    modifier = onnxModifier.from_model_path(input_model_path)
    
    # First fix the pooling pads
    print("\nFixing pooling pads...")
    was_modified = modifier.fix_pooling_pads()
    if was_modified:
        print("Fixed pooling layer paddings")
    
    # Create node states to remove auxiliary branch
    print("\nRemoving auxiliary branch...")
    node_states = {
        'rmse_Output': 'Deleted',  # Remove the RMSE output
        'aux_Output': 'Deleted',   # Remove the auxiliary output
        'regr': 'Deleted'         # Remove the regr input
    }
    
    # Create modification info
    modify_info = {
        'node_states': node_states,
        'added_node_info': {},
        'changed_initializer': {},
        'node_renamed_io': {},
        'added_inputs': {},
        'rebatch_info': {},
        'added_outputs': {},
        'node_changed_attr': {},
        'postprocess_args': {
            'shapeInf': True,  # Run shape inference
            'cleanUp': True    # Remove isolated nodes
        }
    }
    
    # Apply the modifications
    print("Applying modifications...")
    modifier.modify(modify_info)
    
    # Save the modified model
    print("\nSaving modified model...")
    save_path = modifier.check_and_save_model(output_dir)
    print(f"Model saved to: {save_path}")
    
    return save_path

if __name__ == "__main__":
    # Model paths
    input_model_path = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/1-squeezed_FelLumbarVD_Top.onnx"
    output_dir = "/Users/ewern/Desktop/code/MetronMind/onnx-modifier/apr24/fixed_models"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix the model
    fixed_model_path = fix_model(input_model_path, output_dir) 