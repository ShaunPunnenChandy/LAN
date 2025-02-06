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
    
    
