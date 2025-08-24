import torch
import numpy as np
import os
import folder_paths
import time
import cv2
import subprocess
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
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def save_video(self, images, fps, filename_prefix, format, codec, output_dir="", audio=None, save_metadata=True, prompt=None, extra_pnginfo=None):
        base_output_dir = folder_paths.get_output_directory()
        
        if output_dir and output_dir.strip() and output_dir.lower() not in ("false", "none", "null"):
            clean_output_dir = output_dir.lstrip('./\\')
            final_output_dir = os.path.join(base_output_dir, clean_output_dir)
            final_output_dir = os.path.normpath(final_output_dir)
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = base_output_dir
        
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            final_output_path = os.path.join(final_output_dir, filename)
            if not os.path.exists(final_output_path):
                break
            counter += 1
        
        images_np = images.cpu().numpy()
        images_np = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
        
        batch_size, height, width, channels = images_np.shape
        temp_frames_dir = os.path.join(os.path.dirname(final_output_path), "temp_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        try:
            for i, frame_rgb in enumerate(images_np):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            metadata_file_path = None
            if save_metadata:
                video_metadata = gather_comfyui_metadata("DaxNodes VideoSave", prompt, extra_pnginfo)
                if video_metadata:
                    metadata_file_path = create_metadata_file(video_metadata)
            
            if metadata_file_path:
                ffmpeg_cmd = ["ffmpeg", "-y", "-i", metadata_file_path, "-framerate", str(fps), 
                             "-i", os.path.join(temp_frames_dir, "frame_%06d.png"),
                             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                             "-metadata", "creation_time=now", final_output_path]
            else:
                ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", str(fps), 
                             "-i", os.path.join(temp_frames_dir, "frame_%06d.png"),
                             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
                if not save_metadata:
                    ffmpeg_cmd.extend(["-map_metadata", "-1"])
                ffmpeg_cmd.append(final_output_path)
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            
        finally:
            import shutil
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
            cleanup_metadata_file(metadata_file_path)
        
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)
        
        print(f"Saved video: {os.path.basename(final_output_path)} ({file_size:.2f}MB)")
        
        return (final_output_path,)


NODE_CLASS_MAPPINGS = {
    "VideoSave": DaxVideoSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSave": "Video Saver",
}