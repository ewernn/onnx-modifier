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
    
    # Keep only the main output (z_Output_attach_noop_)
    for output in model.graph.output:
        if output.name == 'z_Output_attach_noop_':
            new_model.graph.output.extend([output])
    
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
    
    # Keep all nodes except auxiliary branch nodes
    for node in model.graph.node:
        if node.name not in aux_node_names:
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
        
        # Step 3: Add squeeze/unsqueeze layers
        logger.info("Adding squeeze/unsqueeze layers...")
        mp = onnx.load(output_path)
        mp_fix = onnx.ModelProto()
        mp_fix.CopyFrom(mp)
        mp_fix.graph.ClearField('node')
        mp_fix.graph.ClearField('value_info')
        
        exists = set()
        for i, n in enumerate(mp.graph.node):
            if n.op_type in ['MaxPool', 'AveragePool', "Conv", "BatchNormalization"]:
                if (n.input[0] + '_squeezed' not in exists):
                    mp_fix.graph.node.add().CopyFrom(
                        helper.make_node('Squeeze', 
                                      inputs=[n.input[0]], 
                                      outputs=[n.input[0] + '_squeezed'],
                                      axes=[0]
                        )
                    )
                    
                pool_node = mp_fix.graph.node.add()
                pool_node.CopyFrom(n)
                pool_node.input[0] += '_squeezed'
                pool_node.output[0] += '_before_unsqueeze'
                
                mp_fix.graph.node.add().CopyFrom(
                    helper.make_node('Unsqueeze',
                                  inputs=[pool_node.output[0]],
                                  outputs=[n.output[0]],
                                  axes=[0]
                    )
                )
                
                exists.add(n.input[0] + '_squeezed')
            else:
                mp_fix.graph.node.add().CopyFrom(n)
        
        # Step 4: Run shape inference
        logger.info("Running shape inference...")
        mp_fix = shape_inference.infer_shapes(mp_fix)
        
        # Save squeezed model
        squeezed_path = f"squeezed_{os.path.basename(output_path)}"
        logger.info(f"Saving squeezed model to: {squeezed_path}")
        onnx.save(mp_fix, squeezed_path)
        
        # Step 5: Fix pooling pads
        logger.info("Fixing pooling pads...")
        modifier = onnxModifier.from_model_path(squeezed_path)
        was_modified = modifier.fix_pooling_pads()
        if was_modified:
            logger.info("Fixed pooling layer paddings")
        
        # Step 6: Remove auxiliary branch
        logger.info("Removing auxiliary branch...")
        model = onnx.load(squeezed_path)
        cleaned_model = remove_auxiliary_branch(model)
        
        # Step 7: Save final model
        fixed_path = f"{os.path.splitext(squeezed_path)[0]}_fixed.onnx"
        logger.info(f"Saving final model to: {fixed_path}")
        with open(fixed_path, 'wb') as f:
            f.write(cleaned_model.SerializeToString())
        
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