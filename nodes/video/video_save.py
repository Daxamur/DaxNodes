import torch
import numpy as np
import os
import folder_paths
import time
import cv2
import subprocess
from ...utils.debug_utils import debug_print
from ...utils.metadata_utils import gather_comfyui_metadata, create_metadata_file, cleanup_metadata_file

class DaxVideoSave:
    """Save video with encoding options and inline preview"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "filename_prefix": ("STRING", {
                    "default": "video",
                    "multiline": False
                }),
                "format": (["mp4", "webm", "mov", "avi"],),
                "codec": (["h264", "h265", "vp9", "prores"],),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom output directory (e.g., ./WAN/T2V/). Leave empty for ComfyUI output directory."
                }),
                "audio": ("AUDIO",),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metadata in output video file"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def save_video(self, images, fps, filename_prefix, format, codec, output_dir="", audio=None, save_metadata=True, 
                   prompt=None, extra_pnginfo=None):
        # Always use ComfyUI output directory as base
        base_output_dir = folder_paths.get_output_directory()
        debug_print(f"base_output_dir = {base_output_dir}")
        debug_print(f"output_dir parameter = '{output_dir}'")
        
        # If output_dir specified and not empty/False, create subdirectory within ComfyUI output
        if output_dir and output_dir.strip() and output_dir.lower() not in ("false", "none", "null"):
            # Strip leading slashes and dots to ensure it's a relative path
            clean_output_dir = output_dir.lstrip('./\\')
            final_output_dir = os.path.join(base_output_dir, clean_output_dir)
            # Normalize path to fix any mixed separators
            final_output_dir = os.path.normpath(final_output_dir)
            debug_print(f"cleaned '{output_dir}' -> '{clean_output_dir}'")
            debug_print(f"final_output_dir = {final_output_dir}")
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = base_output_dir
        
        # Generate filename with counter (ComfyUI standard)
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            final_output_path = os.path.join(final_output_dir, filename)
            if not os.path.exists(final_output_path):
                break
            counter += 1
        
        # Convert tensor to numpy
        images_np = images.cpu().numpy()
        images_np = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
        
        # Get dimensions
        batch_size, height, width, channels = images_np.shape
        
        debug_print("Using FFmpeg with H264 for high quality output")
        
        # Use FFmpeg directly for high quality H264 encoding
        temp_frames_dir = os.path.join(os.path.dirname(final_output_path), "temp_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        try:
            # Save frames as individual images
            for i, frame_rgb in enumerate(images_np):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Gather metadata if saving
            metadata_file_path = None
            if save_metadata:
                metadata = gather_comfyui_metadata("DaxNodes VideoSave", prompt, extra_pnginfo)
                if metadata:
                    metadata_file_path = create_metadata_file(metadata)
            
            # Use FFmpeg to create video
            ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", str(fps)]
            
            if metadata_file_path:
                ffmpeg_cmd.extend(["-i", metadata_file_path, "-i", os.path.join(temp_frames_dir, "frame_%06d.png")])
                ffmpeg_cmd.extend(["-map", "1", "-map_metadata", "0"])
            else:
                ffmpeg_cmd.extend(["-i", os.path.join(temp_frames_dir, "frame_%06d.png")])
                if not save_metadata:
                    ffmpeg_cmd.extend(["-map_metadata", "-1"])
            
            ffmpeg_cmd.extend([
                "-c:v", "libx264",
                "-preset", "slow", 
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                final_output_path
            ])
            
            debug_print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            debug_print("FFmpeg success")
            
        finally:
            # Clean up temp frames
            import shutil
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
            
            # Clean up metadata file
            cleanup_metadata_file(metadata_file_path)
        
        # Get file size for info
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        
        print(f"Saved video: {os.path.basename(final_output_path)} ({file_size:.2f}MB)")
        debug_print(f"Format: {format.upper()}, FPS: {fps}, Resolution: {width}x{height}, Codec: {codec}")
        debug_print(f"Full path: {final_output_path}")
        
        # VideoSave only returns the file path
        return (final_output_path,)


NODE_CLASS_MAPPINGS = {
    "VideoSave": DaxVideoSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSave": "Video Saver",
}