import torch
import importlib.util
import os
from pathlib import Path

def get_model():
    path = 'Restormer.basicsr.models.archs.restormer_arch'
    
    # Load the Restormer module dynamically
    module_spec = importlib.util.spec_from_file_location(path, str(Path().joinpath(*path.split('.'))) + '.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if Restormer class exists in module
    if not hasattr(module, "Restormer"):
        raise ImportError("Restormer class not found in the module. Check your model path.")
    
    # Initialize model
    model = module.Restormer(LayerNorm_type='BiasFree').to(device)
    
    # Check for checkpoint
    checkpoint_path = "./checkpoint/real_denoising.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device).get("params", None)
        
        if checkpoint is not None:
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Checkpoint '{checkpoint_path}' loaded successfully.")
        else:
            print(f"⚠️ Warning: Checkpoint '{checkpoint_path}' is empty or invalid. Proceeding without loading.")
    else:
        print(f"⚠️ Warning: Checkpoint file '{checkpoint_path}' not found. Proceeding without loading.")

    model.eval()
    return model
