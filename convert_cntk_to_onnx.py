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

def remove_auxiliary_branch(model):
    """Remove auxiliary branch nodes and clean up inputs/outputs."""
    logger.info("Starting auxiliary branch removal...")
    
    # Create new model
    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.ClearField('node')
    new_model.graph.ClearField('input')
    new_model.graph.ClearField('output')
    
    # Keep only the main input (features)
    for input in model.graph.input:
        if input.name == 'features':
            new_model.graph.input.extend([input])
            logger.info(f"Keeping input: {input.name}")
    
    # Keep only the main output (z_Output_attach_noop_)
    for output in model.graph.output:
        if output.name == 'z_Output_attach_noop_':
            new_model.graph.output.extend([output])
            logger.info(f"Keeping output: {output.name}")
    
    # Keep all initializers
    new_model.graph.ClearField('initializer')
    for init in model.graph.initializer:
        new_model.graph.initializer.extend([init])
    
    # Build a map of node inputs to their nodes
    input_to_node = {}
    for node in model.graph.node:
        for input in node.input:
            if input not in input_to_node:
                input_to_node[input] = []
            input_to_node[input].append(node)
    
    # Find all nodes that are part of the auxiliary branch
    aux_node_names = set()
    nodes_to_process = set()
    
    # Start with nodes that take regr as input
    if 'regr' in input_to_node:
        for node in input_to_node['regr']:
            nodes_to_process.add(node.name)
            logger.info(f"Found auxiliary node: {node.name} (takes regr as input)")
    
    # Process all nodes in the auxiliary branch
    while nodes_to_process:
        node_name = nodes_to_process.pop()
        if node_name in aux_node_names:
            continue
            
        aux_node_names.add(node_name)
        
        # Find the node
        node = None
        for n in model.graph.node:
            if n.name == node_name:
                node = n
                break
                
        if node:
            # Add all nodes that take this node's outputs as input
            for output in node.output:
                if output in input_to_node:
                    for dependent_node in input_to_node[output]:
                        nodes_to_process.add(dependent_node.name)
                        logger.info(f"Found dependent auxiliary node: {dependent_node.name}")
    
    logger.info(f"Found {len(aux_node_names)} auxiliary nodes to remove")
    
    # Keep all nodes except auxiliary branch nodes
    for node in model.graph.node:
        if node.name not in aux_node_names:
            new_model.graph.node.extend([node])
            logger.info(f"Keeping node: {node.name} ({node.op_type})")
        else:
            logger.info(f"Removing node: {node.name} ({node.op_type})")
    
    return new_model

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
        
        # Step 3: Add squeeze/unsqueeze layers more conservatively
        logger.info("Adding squeeze/unsqueeze layers...")
        mp = onnx.load(output_path)
        mp_fix = onnx.ModelProto()
        mp_fix.CopyFrom(mp)
        mp_fix.graph.ClearField('node')
        mp_fix.graph.ClearField('value_info')
        
        # Track which tensors have been squeezed
        squeezed_tensors = set()
        
        for i, n in enumerate(mp.graph.node):
            if n.op_type in ['MaxPool', 'AveragePool', 'Conv']:
                # Only add squeeze/unsqueeze for pooling layers
                input_tensor = n.input[0]
                if input_tensor not in squeezed_tensors:
                    # Add squeeze only if not already squeezed
                    squeeze_node = helper.make_node('Squeeze', 
                                                  inputs=[input_tensor], 
                                                  outputs=[input_tensor + '_squeezed'],
                                                  axes=[0])
                    mp_fix.graph.node.add().CopyFrom(squeeze_node)
                    squeezed_tensors.add(input_tensor)
                
                # Modify pooling node to use squeezed input
                pool_node = mp_fix.graph.node.add()
                pool_node.CopyFrom(n)
                pool_node.input[0] = input_tensor + '_squeezed'
                pool_node.output[0] = n.output[0] + '_before_unsqueeze'
                
                # Add unsqueeze
                unsqueeze_node = helper.make_node('Unsqueeze',
                                                inputs=[pool_node.output[0]],
                                                outputs=[n.output[0]],
                                                axes=[0])
                mp_fix.graph.node.add().CopyFrom(unsqueeze_node)
            else:
                # For non-pooling nodes, check if inputs need to be unsqueezed
                new_node = mp_fix.graph.node.add()
                new_node.CopyFrom(n)
                
                # Check each input
                for j, input_tensor in enumerate(n.input):
                    if input_tensor + '_squeezed' in squeezed_tensors:
                        # This input was squeezed, need to unsqueeze it
                        unsqueeze_node = helper.make_node('Unsqueeze',
                                                        inputs=[input_tensor + '_squeezed'],
                                                        outputs=[input_tensor],
                                                        axes=[0])
                        mp_fix.graph.node.add().CopyFrom(unsqueeze_node)
                        new_node.input[j] = input_tensor
        
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