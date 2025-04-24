import os
import numpy as np
import cntk as C
import onnx
from onnx import helper, shape_inference
from onnx_modifier.onnx_modifier import onnxModifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(model_path):
    """Convert CNTK model to ONNX, keeping only the main output branch."""
    logger.info(f"Converting {model_path}...")
    
    try:
        # Load CNTK model
        logger.info("Loading CNTK model...")
        z = C.Function.load(model_path, device=C.device.cpu())
        
        # Find the main output node (z)
        logger.info("Finding main output node...")
        main_output = z.find_by_name("z", False)
        if main_output is None:
            raise ValueError("Could not find 'z' output node in model")
        
        # Create new model with only main output
        logger.info("Creating new model with only main output...")
        newModel = C.as_composite(main_output)
        
        # Save to ONNX
        output_path = f"{os.path.splitext(model_path)[0]}.onnx"
        logger.info(f"Saving to ONNX format: {output_path}")
        newModel.save(output_path, format=C.ModelFormat.ONNX)
        
        # Fix pooling pads
        logger.info("Fixing pooling pads...")
        modifier = onnxModifier.from_model_path(output_path)
        was_modified = modifier.fix_pooling_pads()
        if was_modified:
            logger.info("Fixed pooling layer paddings")
        
        # Save fixed model
        fixed_path = f"{os.path.splitext(model_path)[0]}_fixed.onnx"
        logger.info(f"Saving fixed model to: {fixed_path}")
        with open(fixed_path, 'wb') as f:
            f.write(modifier.model_proto.SerializeToString())
        
        # Verify the converted model
        logger.info("Verifying converted model...")
        model = onnx.load(fixed_path)
        onnx.checker.check_model(model)
        
        # Print model structure
        logger.info("\n=== Converted Model Structure ===")
        logger.info("\nInputs:")
        for input in model.graph.input:
            logger.info(f"  Name: {input.name}")
            logger.info(f"  Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        
        logger.info("\nOutputs:")
        for output in model.graph.output:
            logger.info(f"  Name: {output.name}")
            logger.info(f"  Shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
        
        return fixed_path
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_cntk_to_onnx.py <path_to_cntk_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    converted_path = convert_model(model_path)
    
    if converted_path:
        print(f"\nConversion successful! Model saved to: {converted_path}")
    else:
        print("\nConversion failed!") 