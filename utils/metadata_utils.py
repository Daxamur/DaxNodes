"""
Metadata utilities for video nodes
"""

import json
import tempfile
import os
from .debug_utils import debug_print

def create_metadata_file(metadata_dict):
    """Create temporary metadata file for FFmpeg"""
    if not metadata_dict:
        return None
        
    metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    try:
        metadata_file.write(";FFMETADATA1\n")
        for key, value in metadata_dict.items():
            # Escape special characters
            escaped_value = str(value).replace("\\", "\\\\").replace(";", "\\;").replace("=", "\\=").replace("\n", "\\n")
            metadata_file.write(f"{key}={escaped_value}\n")
        metadata_file.close()
        return metadata_file.name
    except:
        metadata_file.close()
        os.unlink(metadata_file.name)
        return None

def gather_comfyui_metadata(node_name="DaxNodes", prompt=None, extra_pnginfo=None):
    """Gather ComfyUI prompt and workflow metadata"""
    
    # If prompt and extra_pnginfo are provided directly, use them (preferred)
    if prompt is not None:
        metadata = {
            'prompt': prompt,
            'software': 'ComfyUI',
            'node': node_name
        }
        
        # Add workflow from extra_pnginfo if available
        if extra_pnginfo and 'workflow' in extra_pnginfo:
            metadata['workflow'] = extra_pnginfo['workflow']
        
        return {'comment': json.dumps(metadata)}
    
    # Try to get from execution context
    try:
        import execution
        if hasattr(execution, 'current_prompt') and execution.current_prompt is not None:
            prompt = execution.current_prompt
            extra_data = execution.current_extra_data if hasattr(execution, 'current_extra_data') else {}
            
            metadata = {
                'prompt': prompt,
                'software': 'ComfyUI',
                'node': node_name
            }
            
            if 'extra_pnginfo' in extra_data:
                extra_pnginfo = extra_data['extra_pnginfo']
                if 'workflow' in extra_pnginfo:
                    metadata['workflow'] = extra_pnginfo['workflow']
            
            return {'comment': json.dumps(metadata)}
        
        # Fallback to server instance
        from server import PromptServer
        prompt_server = PromptServer.instance
        if hasattr(prompt_server, 'last_prompt'):
            return {
                'comment': json.dumps({
                    'prompt': prompt_server.last_prompt,
                    'workflow': getattr(prompt_server, 'last_workflow', {}),
                    'software': 'ComfyUI',
                    'node': node_name
                })
            }
    except:
        debug_print("Could not gather ComfyUI metadata")
    return None

def cleanup_metadata_file(metadata_file_path):
    """Clean up temporary metadata file"""
    if metadata_file_path and os.path.exists(metadata_file_path):
        os.unlink(metadata_file_path)