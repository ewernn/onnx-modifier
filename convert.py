import os
import cntk as C
import onnx
from onnx import helper, shape_inference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_onnx(model_path):
    """Convert a CNTK model to ONNX format with explicit softmax pruning check."""
    model_name = os.path.basename(model_path)
    onnx_path = f"{model_path}.onnx"
    
    logger.info(f"\nProcessing {model_name}")
    logger.info("Step 1: Loading CNTK model and checking for CrossEntropyWithSoftmax...")
    
    try:
        # First attempt - direct conversion
        # Use absolute path for loading
        abs_model_path = os.path.abspath(model_path)
        logger.info(f"Loading model from: {abs_model_path}")
        
        z = C.Function.load(abs_model_path, device=C.device.cpu())
        
        # Check if model has CrossEntropyWithSoftmax
        needs_pruning = False
        try:
            pnode = z.find_by_name("aux", False)
            if pnode is not None:
                needs_pruning = True
                logger.info("Found CrossEntropyWithSoftmax layer, will prune it")
            else:
                logger.info("No CrossEntropyWithSoftmax layer found")
        except:
            logger.info("No CrossEntropyWithSoftmax layer found")
        
        if needs_pruning:
            # Prune and convert
            newModel = C.as_composite(pnode)
            newModel.save(onnx_path, format=C.ModelFormat.ONNX)
            logger.info(f"Converted with pruning: {onnx_path}")
        else:
            # Direct conversion
            z.save(onnx_path, format=C.ModelFormat.ONNX)
            logger.info(f"Converted directly: {onnx_path}")
            
    except Exception as e:
        logger.error(f"Failed to convert {model_name}")
        logger.error(str(e))
        return False
    
    return True

def create_squeezed_version(model_path):
    """Create a squeezed version of an ONNX model."""
    onnx_path = f"{model_path}.onnx"
    squeezed_path = f"squeezed_{os.path.basename(onnx_path)}"
    
    logger.info(f"Step 2: Creating squeezed version: {squeezed_path}")
    
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
        logger.info(f"Successfully created squeezed version")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create squeezed version for {model_path}")
        logger.error(str(e))
        return False

def main():
    # Get the current working directory
    base_dir = os.getcwd()
    logger.info(f"Working directory: {base_dir}")
    logger.info(f"Working directory: {'/home/ewern/onnx-modifier'}")
    
    # List of models to convert
    models = [
        'CanCTR_0_0', 'CanCTR_1_0', 'CanCTR_2_0', 'CanCTR_3_0', 'CanCTR_4_0', 'CanCTR_Top',
        'CanElbowBad', 'CanShoulderConds', 'CanShoulderPnts_top', 'EqFootOblique',
        'FelLumbarVD_0_0', 'FelLumbarVD_1_0', 'FelLumbarVD_2_0', 'FelLumbarVD_3_0',
        'FelLumbarVD_4_0', 'FelLumbarVD_5_0', 'FelLumbarVD_6_0', 'FelLumbarVD_7_0',
        'FelLumbarVD_Locator', 'FelLumbarVD_Top'
    ]
    
    logger.info(f"Found {len(models)} CNTK models to convert")
    
    # Verify files exist before processing
    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {model}")
        
        # Remove existing ONNX files if they exist
        if os.path.exists(f"{model}.onnx"):
            os.remove(f"{model}.onnx")
        if os.path.exists(f"squeezed_{model}.onnx"):
            os.remove(f"squeezed_{model}.onnx")
            
        # Step 1: Convert to ONNX
        if convert_to_onnx(model_path):
            # Step 2: Create squeezed version
            create_squeezed_version(model_path)
        
        logger.info(f"Completed processing {model}")
        logger.info('='*50)

if __name__ == "__main__":
    main()