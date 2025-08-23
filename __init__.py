"""
DaxNodes - Professional ComfyUI Node Pack
A collection of high-quality video processing and utility nodes for ComfyUI
"""

import os
import sys
import subprocess
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Dax"

def install_requirements():
    """Auto-install required dependencies"""
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        return
    
    missing_deps = []
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        missing_deps.append("color-matcher")
    
    if missing_deps:
        print(f"[DaxNodes] Installing dependencies: {', '.join(missing_deps)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[DaxNodes] Dependencies installed successfully")
        except:
            print(f"[DaxNodes] Please manually install: pip install -r {requirements_file}")

# Auto-install dependencies
install_requirements()

# Import node modules
from .nodes.video.video_segment_saver_v2 import NODE_CLASS_MAPPINGS as SAVER_NODES, NODE_DISPLAY_NAME_MAPPINGS as SAVER_NAMES
from .nodes.video.video_segment_combiner_v2 import NODE_CLASS_MAPPINGS as COMBINER_NODES, NODE_DISPLAY_NAME_MAPPINGS as COMBINER_NAMES
from .nodes.video.video_preview import NODE_CLASS_MAPPINGS as PREVIEW_NODES, NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NAMES
from .nodes.video.video_save import NODE_CLASS_MAPPINGS as SAVE_NODES, NODE_DISPLAY_NAME_MAPPINGS as SAVE_NAMES
from .nodes.video.video_stream_upscaler import NODE_CLASS_MAPPINGS as UPSCALER_NODES, NODE_DISPLAY_NAME_MAPPINGS as UPSCALER_NAMES
from .nodes.video.video_stream_rife_vfi import NODE_CLASS_MAPPINGS as RIFE_NODES, NODE_DISPLAY_NAME_MAPPINGS as RIFE_NAMES
from .nodes.video.video_color_correct_v3 import NODE_CLASS_MAPPINGS as COLOR_NODES, NODE_DISPLAY_NAME_MAPPINGS as COLOR_NAMES
from .nodes.video.face_frame_detector import NODE_CLASS_MAPPINGS as FACE_NODES, NODE_DISPLAY_NAME_MAPPINGS as FACE_NAMES

from .nodes.utility.string_nodes import NODE_CLASS_MAPPINGS as STRING_NODES, NODE_DISPLAY_NAME_MAPPINGS as STRING_NAMES
from .nodes.utility.wan_resolution_nodes import NODE_CLASS_MAPPINGS as RESOLUTION_NODES, NODE_DISPLAY_NAME_MAPPINGS as RESOLUTION_NAMES
from .nodes.utility.batch_utils import NODE_CLASS_MAPPINGS as BATCH_NODES, NODE_DISPLAY_NAME_MAPPINGS as BATCH_NAMES
from .nodes.utility.runtime_generation_settings import NODE_CLASS_MAPPINGS as RUNTIME_NODES, NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NAMES

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **SAVER_NODES, **COMBINER_NODES, **PREVIEW_NODES, **SAVE_NODES,
    **UPSCALER_NODES, **RIFE_NODES, **COLOR_NODES, **FACE_NODES,
    **STRING_NODES, **RESOLUTION_NODES, **BATCH_NODES, **RUNTIME_NODES
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SAVER_NAMES, **COMBINER_NAMES, **PREVIEW_NAMES, **SAVE_NAMES,
    **UPSCALER_NAMES, **RIFE_NAMES, **COLOR_NAMES, **FACE_NAMES,
    **STRING_NAMES, **RESOLUTION_NAMES, **BATCH_NAMES, **RUNTIME_NAMES
}

# JavaScript extensions directory
WEB_DIRECTORY = "./web/comfyui"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]