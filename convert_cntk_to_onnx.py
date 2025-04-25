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

def fix_pooling_pads(model):
    """Fix pooling layer padding issues in the ONNX model."""
    # Create new model
    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.ClearField('node')
    
    # Process each node
    for node in model.graph.node:
        if node.op_type in ['MaxPool', 'AveragePool']:
            # Create new node with fixed padding
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            
            # Find the pads attribute
            pads_attr = None
            for attr in new_node.attribute:
                if attr.name == 'pads':
                    pads_attr = attr
                    break
            
            if pads_attr:
                # Fix padding to be symmetric and correct size
                pads = list(pads_attr.ints)
                if len(pads) == 6:  # 3D pooling format
                    # Convert to 2D pooling format [top, left, bottom, right]
                    pads = [pads[1], pads[2], pads[4], pads[5]]
                elif len(pads) == 4:  # 2D pooling
                    # Make sure padding is symmetric
                    pads[0] = pads[2] = max(pads[0], pads[2])
                    pads[1] = pads[3] = max(pads[1], pads[3])
                pads_attr.ints[:] = pads
            
            new_model.graph.node.extend([new_node])
        else:
            new_model.graph.node.extend([node])
    
    return new_model

def convert_model(model_path):
    """Convert CNTK model to ONNX with all necessary steps."""
    logger.info(f"Converting {model_path}...")
    
    try:
        # Step 1: Load CNTK model
        logger.info("Loading CNTK model...")
        z = C.Function.load(model_path, device=C.device.cpu())
        
        # Step 2: Try direct conversion first
        logger.info("Attempting direct conversion...")
        output_path = f"{os.path.splitext(model_path)[0]}.onnx"
        try:
            z.save(output_path, format=C.ModelFormat.ONNX)
        except Exception as e:
            logger.info(f"Direct conversion failed: {e}")
            logger.info("Attempting to prune CrossEntropyWithSoftmax layer...")
            # Find and prune the aux node
            pnode = z.find_by_name("aux", False)
            if pnode is None:
                raise ValueError("Could not find 'aux' node for pruning")
            newModel = C.as_composite(pnode)
            newModel.save(output_path, format=C.ModelFormat.ONNX)
        
        # Step 3: Add single squeeze at input and unsqueeze at output
        logger.info("Adding single squeeze at input and unsqueeze at output...")
        mp = onnx.load(output_path)
        mp_fix = onnx.ModelProto()
        mp_fix.CopyFrom(mp)
        mp_fix.graph.ClearField('node')
        mp_fix.graph.ClearField('value_info')
        
        # Build a map of all tensors and their producer/consumer nodes
        tensor_producers = {}
        tensor_consumers = {}
        
        for node in mp.graph.node:
            for output in node.output:
                tensor_producers[output] = node
            
            for input in node.input:
                if input not in tensor_consumers:
                    tensor_consumers[input] = []
                tensor_consumers[input].append(node)
        
        # Find model inputs and outputs
        model_inputs = [input.name for input in mp.graph.input]
        model_outputs = [output.name for output in mp.graph.output]
        
        # Add nodes to the new model with squeeze at beginning and unsqueeze at end
        added_nodes = set()
        processed_tensors = set()
        
        # Add squeeze nodes for model inputs (except 'regr' input)
        for input_name in model_inputs:
            if input_name != 'regr':
                # Add squeeze for this input
                squeeze_node = helper.make_node(
                    'Squeeze',
                    inputs=[input_name],
                    outputs=[input_name + '_squeezed'],
                    axes=[0]
                )
                mp_fix.graph.node.extend([squeeze_node])
                processed_tensors.add(input_name)
                logger.info(f"Added squeeze node for input: {input_name}")
        
        # Process all other nodes
        for node in mp.graph.node:
            # Skip nodes that have already been added
            if node.name in added_nodes:
                continue
            
            # Create a copy of the node
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            
            # Update input references if they have been squeezed
            for i, input_name in enumerate(new_node.input):
                # Skip initializers and control inputs
                if input_name not in model_inputs and input_name not in tensor_producers:
                    continue
                
                if input_name in processed_tensors:
                    # This input has been processed, use its squeezed version
                    new_node.input[i] = input_name + '_squeezed'
            
            # Update outputs to be squeezed versions
            for i, output_name in enumerate(new_node.output):
                if output_name in model_outputs:
                    # If this is a model output, keep original name and add unsqueeze later
                    continue
                new_node.output[i] = output_name + '_squeezed'
                processed_tensors.add(output_name)
            
            # Add the modified node
            mp_fix.graph.node.extend([new_node])
            added_nodes.add(node.name)
        
        # Add unsqueeze nodes for model outputs
        for output_name in model_outputs:
            # Find the node that produces this output
            if output_name in tensor_producers:
                producer = tensor_producers[output_name]
                producer_idx = mp.graph.node.index(producer)
                
                # Get the squeezed version of this output
                squeezed_output = output_name + '_squeezed'
                
                # Add unsqueeze for model output
                unsqueeze_node = helper.make_node(
                    'Unsqueeze',
                    inputs=[squeezed_output],
                    outputs=[output_name],
                    axes=[0]
                )
                mp_fix.graph.node.extend([unsqueeze_node])
                logger.info(f"Added unsqueeze node for output: {output_name}")
        
        # Run shape inference
        logger.info("Running shape inference...")
        mp_fix = shape_inference.infer_shapes(mp_fix)
        
        # Save squeezed model
        squeezed_path = f"squeezed_{os.path.basename(output_path)}"
        logger.info(f"Saving squeezed model to: {squeezed_path}")
        onnx.save(mp_fix, squeezed_path)
        
        # Step 4: Fix pooling pads
        logger.info("Fixing pooling pads...")
        model = onnx.load(squeezed_path)
        fixed_model = fix_pooling_pads(model)
        
        # Save the final model
        fixed_path = f"{os.path.splitext(squeezed_path)[0]}_fixed.onnx"
        logger.info(f"Saving model with fixed pooling pads to: {fixed_path}")
        with open(fixed_path, 'wb') as f:
            f.write(fixed_model.SerializeToString())
        
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