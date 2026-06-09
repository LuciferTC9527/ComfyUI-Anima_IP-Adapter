"""
Anima IP-Adapter ComfyUI Custom Node (SigLIP2).
"""

import os
import folder_paths

# Register model folders
models_dir = folder_paths.models_dir
folder_paths.folder_names_and_paths.setdefault("ipadapter", (
    [os.path.join(models_dir, "ipadapter")],
    {".safetensors"},
))
SIGLIP2_DIR = os.path.join(models_dir, "siglip2", "siglip2-base-patch16-512")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
