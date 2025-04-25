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
        
        # Step 3: Add squeeze/unsqueeze layers with look-ahead
        logger.info("Adding squeeze/unsqueeze layers with look-ahead...")
        mp = onnx.load(output_path)
        mp_fix = onnx.ModelProto()
        mp_fix.CopyFrom(mp)
        mp_fix.graph.ClearField('node')
        mp_fix.graph.ClearField('value_info')
        
        # Define target operations that need squeezing
        target_ops = ['MaxPool', 'AveragePool', 'Conv', 'BatchNormalization']
        
        # Build a map of tensor outputs to their consumer nodes
        output_to_consumers = {}
        for node in mp.graph.node:
            for input_name in node.input:
                if input_name not in output_to_consumers:
                    output_to_consumers[input_name] = []
                output_to_consumers[input_name].append(node)
        
        # Map of original tensor names to their squeezed versions
        tensor_to_squeezed = {}
        # Track tensors we've already processed
        processed_tensors = set()
        
        # First pass: Add nodes to the model
        added_nodes = []
        for i, node in enumerate(mp.graph.node):
            is_target_op = node.op_type in target_ops
            
            if is_target_op:
                # This operation needs squeezed input
                input_tensor = node.input[0]
                
                # Check if the input is already in squeezed form
                if input_tensor in tensor_to_squeezed:
                    # Input already has a squeezed version
                    squeezed_input = tensor_to_squeezed[input_tensor]
                else:
                    # Need to add squeeze operation
                    squeezed_input = input_tensor + '_squeezed'
                    tensor_to_squeezed[input_tensor] = squeezed_input
                    
                    if squeezed_input not in processed_tensors:
                        # Add the squeeze operation
                        squeeze_node = helper.make_node('Squeeze', 
                                                    inputs=[input_tensor], 
                                                    outputs=[squeezed_input],
                                                    axes=[0])
                        added_nodes.append(squeeze_node)
                        processed_tensors.add(squeezed_input)
                
                # Create a copy of the target operation with squeezed input
                modified_node = onnx.NodeProto()
                modified_node.CopyFrom(node)
                modified_node.input[0] = squeezed_input
                
                # Determine if output should remain squeezed
                output_tensor = node.output[0]
                consumers = output_to_consumers.get(output_tensor, [])
                should_unsqueeze = True
                
                # Check if ALL consumers need squeezed input
                if consumers:
                    should_unsqueeze = False
                    for consumer in consumers:
                        if consumer.op_type not in target_ops:
                            should_unsqueeze = True
                            break
                
                if should_unsqueeze:
                    # We need to unsqueeze before passing to non-target consumers
                    squeezed_output = output_tensor + '_before_unsqueeze'
                    modified_node.output[0] = squeezed_output
                    
                    # Add unsqueeze operation
                    unsqueeze_node = helper.make_node('Unsqueeze',
                                                  inputs=[squeezed_output],
                                                  outputs=[output_tensor],
                                                  axes=[0])
                    added_nodes.append(modified_node)
                    added_nodes.append(unsqueeze_node)
                else:
                    # Keep output in squeezed form for next target operation
                    squeezed_output = output_tensor + '_squeezed'
                    modified_node.output[0] = squeezed_output
                    tensor_to_squeezed[output_tensor] = squeezed_output
                    added_nodes.append(modified_node)
            else:
                # Regular non-target node
                modified_node = onnx.NodeProto()
                modified_node.CopyFrom(node)
                
                # Check each input to see if it needs unsqueezing
                for j, input_name in enumerate(node.input):
                    if input_name in tensor_to_squeezed:
                        # This input has a squeezed version, so we need to unsqueeze
                        squeezed_input = tensor_to_squeezed[input_name]
                        
                        # Add unsqueeze if not already done
                        if input_name not in processed_tensors:
                            unsqueeze_node = helper.make_node('Unsqueeze',
                                                        inputs=[squeezed_input],
                                                        outputs=[input_name],
                                                        axes=[0])
                            added_nodes.append(unsqueeze_node)
                            processed_tensors.add(input_name)
                
                added_nodes.append(modified_node)
        
        # Add all nodes to the model in the order we processed them
        for node in added_nodes:
            mp_fix.graph.node.add().CopyFrom(node)
        
        # Step 4: Run shape inference
        logger.info("Running shape inference...")
        mp_fix = shape_inference.infer_shapes(mp_fix)
        
        # Save squeezed model
        squeezed_path = f"squeezed_{os.path.basename(output_path)}"
        logger.info(f"Saving squeezed model to: {squeezed_path}")
        onnx.save(mp_fix, squeezed_path)
        
        # Step 5: Fix pooling pads
        logger.info("Fixing pooling pads...")
        model = onnx.load(squeezed_path)
        fixed_model = fix_pooling_pads(model)
        
        # Save the model with fixed pooling pads
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