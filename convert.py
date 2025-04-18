import os
import cntk as C
import onnx
from onnx import helper, shape_inference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_onnx(model_path):
    """Convert a CNTK model to ONNX format."""
    model_name = os.path.basename(model_path)
    onnx_path = f"{model_path}.onnx"
    
    logger.info(f"Converting {model_name} to ONNX...")
    
    try:
        # First attempt - direct conversion
        z = C.Function.load(model_path, device=C.device.cpu())
        z.save(onnx_path, format=C.ModelFormat.ONNX)
        logger.info(f"Direct conversion successful: {onnx_path}")
        
    except Exception as e:
        logger.info(f"Direct conversion failed, attempting to prune CrossEntropyWithSoftmax...")
        try:
            # Try to find and prune CrossEntropyWithSoftmax
            pnode = z.find_by_name("aux", False)
            if pnode is None:
                raise ValueError("Could not find 'aux' node")
            newModel = C.as_composite(pnode)
            newModel.save(onnx_path, format=C.ModelFormat.ONNX)
            logger.info(f"Pruned conversion successful: {onnx_path}")
        except Exception as e2:
            logger.error(f"Both conversion attempts failed for {model_name}")
            logger.error(f"Error 1: {str(e)}")
            logger.error(f"Error 2: {str(e2)}")
            return False
    
    return True

def create_squeezed_version(model_path):
    """Create a squeezed version of an ONNX model."""
    onnx_path = f"{model_path}.onnx"
    squeezed_path = f"squeezed_{os.path.basename(onnx_path)}"
    
    logger.info(f"Creating squeezed version: {squeezed_path}")
    
    try:
        mp = onnx.load(onnx_path)
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
                                      axes=[2]))
                    pool_node = mp_fix.graph.node.add()
                    pool_node.CopyFrom(n)
                    pool_node.input[0] += '_squeezed'
                    pool_node.output[0] += '_before_unsqueeze'
                    mp_fix.graph.node.add().CopyFrom(
                        helper.make_node('Unsqueeze', 
                                      inputs=[pool_node.output[0]], 
                                      outputs=[n.output[0]], 
                                      axes=[2]))
                    exists.add(n.input[0] + '_squeezed')
            else:
                mp_fix.graph.node.add().CopyFrom(n)
        
        mp_fix = shape_inference.infer_shapes(mp_fix)
        onnx.save(mp_fix, squeezed_path)
        logger.info(f"Successfully created squeezed version: {squeezed_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create squeezed version for {model_path}")
        logger.error(str(e))
        return False

def main():
    # Get list of CNTK models (files without extension)
    models = [f for f in os.listdir('.') if os.path.isfile(f) and '.' not in f]
    
    logger.info(f"Found {len(models)} CNTK models to convert")
    
    for model in models:
        logger.info(f"\nProcessing {model}")
        
        # Step 1: Convert to ONNX
        if convert_to_onnx(model):
            # Step 2: Create squeezed version
            create_squeezed_version(model)
        
        logger.info(f"Completed processing {model}\n")

if __name__ == "__main__":
    main()