"""
Metadata utilities for video nodes
"""

import json
import tempfile
import os

def create_metadata_file(video_metadata):
    """Create temporary metadata file for FFmpeg"""
    if not video_metadata:
        return None
    metadata = json.dumps(video_metadata)
    metadata = metadata.replace("\\","\\\\")
    metadata = metadata.replace(";","\\;")
    metadata = metadata.replace("#","\\#")
    metadata = metadata.replace("=","\\=")
    metadata = metadata.replace("\n","\\\n")
    metadata = "comment=" + metadata
    
    metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    try:
        metadata_file.write(";FFMETADATA1\n")
        metadata_file.write(metadata)
        metadata_file.close()
        return metadata_file.name
    except:
        metadata_file.close()
        os.unlink(metadata_file.name)
        return None

def gather_comfyui_metadata(node_name="DaxNodes", prompt=None, extra_pnginfo=None):
    """Gather ComfyUI prompt and workflow metadata"""
    
    if prompt is not None or extra_pnginfo is not None:
        video_metadata = {}
        
        if prompt is not None:
            video_metadata["prompt"] = json.dumps(prompt)
        
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                video_metadata[x] = extra_pnginfo[x]
        import datetime
        video_metadata["CreationTime"] = datetime.datetime.now().isoformat(" ")[:19]
        video_metadata["software"] = "ComfyUI"
        video_metadata["node"] = node_name
        
        return video_metadata
    
    return None

def cleanup_metadata_file(metadata_file_path):
    """Clean up temporary metadata file"""
    if metadata_file_path and os.path.exists(metadata_file_path):
        os.unlink(metadata_file_path)