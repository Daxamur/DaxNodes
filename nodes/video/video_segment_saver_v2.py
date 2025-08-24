import torch
import numpy as np
import os
import folder_paths
import time
import cv2
import json
import subprocess
from ...utils.debug_utils import debug_print

class DaxVideoSegmentSaver:
    """Video segment saver with caching and state management
    
    Efficiently saves video segments with execution tracking to prevent duplicate processing.
    Features intelligent caching and support for multiple video formats.
    """
    
    # Class-level storage for tracking executions
    _execution_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "execution_id": ("INT", {
                    "default": 0,
                    "tooltip": "Unique ID for this execution run (changes each run)"
                }),
                "loop_index": ("INT", {
                    "default": 0,
                    "tooltip": "Current loop iteration number"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "filename_prefix": ("STRING", {
                    "default": "segment",
                    "multiline": False
                }),
                "format": (["mp4", "webm", "mov", "avi"],),
                "codec": (["h264", "h265", "vp9", "prores"],),
            },
            "optional": {
                "se_result": ("INT", {
                    "tooltip": "Skip save when equals 1, save when any other value"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("packed_filenames", "execution_id")
    FUNCTION = "save_segment"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def get_state_key(self, execution_id, filename_prefix):
        """Generate a unique key for this execution context"""
        return f"{execution_id}_{filename_prefix}"
    
    def save_segment(self, images, execution_id, loop_index, 
                           fps, filename_prefix, format, codec, se_result=None):
        
        # Check se_result for conditional saving
        if se_result == 1:
            print(f"Skipping save: se_result={se_result} (execution={execution_id}, loop={loop_index})")
            return ("", execution_id)
        
        state_key = self.get_state_key(execution_id, filename_prefix)
        
        print(f"Saving segment: Exec={execution_id}, Loop={loop_index}")
        debug_print(f"State key: {state_key}")
        debug_print(f"Input images shape: {images.shape}")
        
        # Initialize execution cache if needed
        if state_key not in self._execution_cache:
            self._execution_cache[state_key] = {
                "execution_id": execution_id,
                "segments": {},  # Dict of loop_index -> filepath
                "last_packed": "",
                "timestamp": time.time()
            }
            debug_print(f"Initialized new execution context: {state_key}")
        
        cache = self._execution_cache[state_key]
        
        # Check if this is a new execution (execution_id changed)
        if cache["execution_id"] != execution_id:
            debug_print(f"New execution detected (old={cache['execution_id']}, new={execution_id})")
            # Clear the cache for this new execution
            cache["execution_id"] = execution_id
            cache["segments"] = {}
            cache["last_packed"] = ""
            cache["timestamp"] = time.time()
        
        # Create structured directory: output/.tmp/<execution_id>/
        base_output_dir = folder_paths.get_output_directory()
        tmp_base_dir = os.path.join(base_output_dir, ".tmp")
        execution_dir = os.path.join(tmp_base_dir, str(execution_id))
        
        # Create the directory structure
        os.makedirs(execution_dir, exist_ok=True)
        
        debug_print(f"Segment directory: {execution_dir}")
        
        # Generate filename for this segment: <executionid>_<loop_index>.mp4
        filename = f"{execution_id}_{loop_index:05d}.{format}"
        filepath = os.path.join(execution_dir, filename)
        
        # Check if we're replacing an existing segment
        if loop_index in cache["segments"]:
            old_filepath = cache["segments"][loop_index]
            if os.path.exists(old_filepath) and old_filepath != filepath:
                try:
                    os.remove(old_filepath)
                    debug_print(f"Removed old segment for loop {loop_index}: {old_filepath}")
                except:
                    pass
        
        # Convert tensor to numpy
        images_np = images.cpu().numpy()
        images_np = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
        
        # Get dimensions
        batch_size, height, width, channels = images_np.shape
        
        debug_print(f"Saving segment: {filename} ({batch_size} frames)")
        
        codec_settings = {
            "h264": ["-c:v", "libx264", "-preset", "slow", "-crf", "18"],
            "h265": ["-c:v", "libx265", "-preset", "slow", "-crf", "18"],
            "vp9": ["-c:v", "libvpx-vp9", "-crf", "18"],
            "prores": ["-c:v", "prores_ks", "-profile:v", "3"]
        }
        
        debug_print(f"Using FFmpeg with {codec} codec for high quality")
        
        temp_frames_dir = os.path.join(execution_dir, "temp_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        try:
            # Save frames as individual images
            frame_paths = []
            for i, frame_rgb in enumerate(images_np):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            
            # Build FFmpeg command with selected codec
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_frames_dir, "frame_%06d.png"),
            ] + codec_settings.get(codec, codec_settings["h264"]) + [
                "-pix_fmt", "yuv420p",
                filepath
            ]
            
            debug_print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            debug_print(f"FFmpeg success: {result.stdout}")
            
        finally:
            # Clean up temp frames
            import shutil
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
        
        # Update cache with this segment
        cache["segments"][loop_index] = filepath
        
        # Rebuild packed filenames from segments in order
        sorted_indices = sorted(cache["segments"].keys())
        segment_paths = [cache["segments"][idx] for idx in sorted_indices]
        new_packed = "<<<SPLIT>>>".join(segment_paths)
        cache["last_packed"] = new_packed
        
        # Save state to disk for persistence (same directory as segments)
        state_file = os.path.join(execution_dir, f"segment_state_{state_key}.json")
        with open(state_file, 'w') as f:
            json.dump({
                "execution_id": execution_id,
                "segments": cache["segments"],
                "last_packed": new_packed,
                "timestamp": cache["timestamp"]
            }, f)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Segment saved: {os.path.basename(filepath)} ({file_size:.2f}MB)")
        debug_print(f"Total segments in execution: {len(cache['segments'])}")
        
        return (new_packed, execution_id)


class VideoSegmentStateLoader:
    """Load saved segment state from a previous execution"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "execution_id": ("INT", {
                    "default": 0,
                    "tooltip": "Execution ID to load state for"
                }),
                "filename_prefix": ("STRING", {
                    "default": "segment",
                    "tooltip": "Filename prefix used in the saver"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("packed_filenames",)
    FUNCTION = "load_state"
    CATEGORY = "Video"
    
    def load_state(self, execution_id, filename_prefix):
        state_key = f"{execution_id}_{filename_prefix}"
        state_file = os.path.join(folder_paths.get_temp_directory(), f"segment_state_{state_key}.json")
        
        debug_print(f"Loading state for {state_key}")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                packed = state.get("last_packed", "")
                segments = state.get("segments", {})
                debug_print(f"Loaded {len(segments)} segments from previous execution")
                return (packed,)
            except Exception as e:
                print(f"Error loading state: {e}")
                return ("",)
        else:
            debug_print("No saved state found")
            return ("",)


NODE_CLASS_MAPPINGS = {
    "VideoSegmentSaverV2": DaxVideoSegmentSaver,
    "VideoSegmentStateLoader": VideoSegmentStateLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSegmentSaverV2": "Video Segment Saver",
    "VideoSegmentStateLoader": "Video Segment State Loader",
}