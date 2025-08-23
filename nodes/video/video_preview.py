import os
import shutil
import folder_paths
from ...utils.debug_utils import debug_print

class VideoPreview:
    """Preview video from filepath in ComfyUI interface"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "tooltip": "Path to video file to preview"
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def preview_video(self, video_path):
        debug_print(f"Input path: {video_path}")
        
        output_dir = folder_paths.get_output_directory()
        
        # Check if it's an absolute path
        if os.path.isabs(video_path):
            # Check if file is already in output directory
            if video_path.startswith(output_dir):
                # File is already in output dir, just use the relative part
                filename = os.path.relpath(video_path, output_dir)
                display_path = video_path
                debug_print(f"File already in output directory: {filename}")
            elif os.path.exists(video_path):
                # File exists elsewhere, copy to output
                filename = os.path.basename(video_path)
                display_path = os.path.join(output_dir, filename)
                if video_path != display_path:
                    debug_print(f"Copying {video_path} to {display_path}")
                    shutil.copy2(video_path, display_path)
            else:
                raise ValueError(f"Video file not found: {video_path}")
        else:
            # Relative path or just filename
            filename = video_path
            display_path = os.path.join(output_dir, filename)
            if not os.path.exists(display_path):
                raise ValueError(f"Video file not found: {display_path}")
        
        # Normalize filename for ComfyUI (remove any subdirectory paths)
        if '\\' in filename:
            filename = filename.replace('\\', '/')
        
        # Split subdirectory and filename for proper ComfyUI handling
        if '/' in filename:
            # Extract subfolder and base filename
            subfolder = os.path.dirname(filename)
            base_filename = os.path.basename(filename)
            debug_print(f"Subfolder: {subfolder}")
            debug_print(f"Base filename: {base_filename}")
        else:
            subfolder = ""
            base_filename = filename
        
        # Get file info
        file_size = os.path.getsize(display_path) / (1024 * 1024)  # MB
        debug_print(f"File size: {file_size:.2f}MB")
        debug_print(f"Display filename: {filename}")
        
        # Get video format from file extension
        _, ext = os.path.splitext(filename)
        format_type = ext[1:].lower() if ext else "mp4"  # Remove the dot
        
        # Try to detect frame rate using ffprobe
        frame_rate = 30.0  # Default fallback
        try:
            import subprocess
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_streams", "-select_streams", "v:0", display_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if data.get("streams"):
                    fps_str = data["streams"][0].get("r_frame_rate", "30/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        frame_rate = float(num) / float(den)
                    else:
                        frame_rate = float(fps_str)
                    debug_print(f"Detected frame rate: {frame_rate}")
        except Exception as e:
            debug_print(f"Could not detect frame rate, using default: {e}")
        
        # Return video for display
        result = {
            "filename": base_filename,  # Just the filename without subdirectory
            "subfolder": subfolder,  # Subdirectory path (if any)
            "type": "output",
            "format": f"video/h264-{format_type}",  # e.g., "video/h264-mp4"
            "frame_rate": frame_rate,
            "workflow": filename,  # VideoHelperSuite uses metadata PNG, we'll use video file
            "fullpath": display_path,  # Full path to actual video file
        }
        
        debug_print(f"Preview result: {result}")
        return {"ui": {"gifs": [result]}}


NODE_CLASS_MAPPINGS = {
    "VideoPreview": VideoPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPreview": "Video Preview",
}