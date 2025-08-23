import torch
import numpy as np
import os
import folder_paths
import time
import cv2
import json
import subprocess
from ...utils.debug_utils import debug_print

class DaxVideoSegmentCombinerV2:
    """Combine video segments using execution ID to find segments automatically"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "execution_id": ("INT", {
                    "default": 0,
                    "tooltip": "Execution ID to combine segments for"
                }),
                "output_prefix": ("STRING", {
                    "default": "combined_video",
                    "tooltip": "Prefix for the combined output video"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "format": (["mp4", "webm", "mov", "avi"],),
                "codec": (["h264", "h265", "vp9", "prores"],),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom output directory. Leave empty for ComfyUI output."
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metadata in output video file"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_filepath", "preview_frames")
    FUNCTION = "combine_segments_v2"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def load_segments_from_directory(self, execution_id):
        """Scan execution directory and find all video segments"""
        # Look in structured directory: output/.tmp/<execution_id>/
        base_output_dir = folder_paths.get_output_directory()
        tmp_base_dir = os.path.join(base_output_dir, ".tmp")
        execution_dir = os.path.join(tmp_base_dir, str(execution_id))
        
        debug_print(f"Scanning execution directory: {execution_dir}")
        
        if not os.path.exists(execution_dir):
            debug_print(f"Execution directory does not exist: {execution_dir}")
            # Check what execution dirs DO exist
            if os.path.exists(tmp_base_dir):
                existing_dirs = [d for d in os.listdir(tmp_base_dir) if os.path.isdir(os.path.join(tmp_base_dir, d))]
                debug_print(f"Available execution directories in .tmp: {existing_dirs}")
            return {}
        
        # Find all video files that match pattern: <execution_id>_<loop_index>.ext
        segments = {}
        files = os.listdir(execution_dir)
        
        for file in files:
            # Look for video files matching pattern: <execution_id>_<loop_index>.ext
            if file.endswith(('.mp4', '.webm', '.mov', '.avi')):
                try:
                    # Parse filename: execution_id_loop_index.ext
                    name_part = file.split('.')[0]  # Remove extension
                    parts = name_part.split('_')
                    
                    if len(parts) >= 2:
                        file_exec_id = parts[0]
                        loop_index_str = parts[1]
                        
                        # Verify this file belongs to our execution
                        if file_exec_id == str(execution_id):
                            try:
                                loop_index = int(loop_index_str)
                                file_path = os.path.join(execution_dir, file)
                                segments[loop_index] = file_path
                                debug_print(f"Found segment {loop_index}: {file}")
                            except ValueError:
                                debug_print(f"Skipping file with invalid loop index: {file}")
                        else:
                            debug_print(f"Skipping file for different execution: {file}")
                except Exception as e:
                    debug_print(f"Error parsing filename {file}: {e}")
        
        debug_print(f"Found {len(segments)} segments: {sorted(segments.keys())}")
        return segments
    
    def extract_frames_from_video(self, video_path, num_frames=1, from_end=False):
        """Extract specific frames from a video file using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            if from_end:
                # Get last frame(s)
                start_frame = max(0, total_frames - num_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(min(num_frames, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to 0-1 range
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
            
            return frames if len(frames) > 1 else (frames[0] if frames else None)
        finally:
            cap.release()
    
    
    def combine_segments_v2(self, execution_id, output_prefix, fps, format, codec, output_dir="", save_metadata=True):
        
        print(f"Combining segments for execution {execution_id}")
        debug_print(f"Input parameters: execution_id={execution_id}, output_prefix='{output_prefix}', fps={fps}, format={format}, codec={codec}, output_dir='{output_dir}'")
        
        # Load segments by scanning the execution directory
        debug_print("Loading segments from execution directory...")
        segments = self.load_segments_from_directory(execution_id)
        
        if not segments:
            print(f"ERROR: No segments found for execution {execution_id}")
            raise ValueError(f"No video segments found in execution directory for execution_id={execution_id}")
        
        debug_print(f"Raw segments dictionary: {segments}")
        
        # Sort segments by loop index
        sorted_indices = sorted(segments.keys())
        segment_paths = [segments[idx] for idx in sorted_indices]
        
        print(f"Found {len(segment_paths)} segments to combine")
        debug_print(f"Sorted indices: {sorted_indices}")
        for idx, path in enumerate(segment_paths):
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                debug_print(f"Segment {idx}: {os.path.basename(path)} ({size_mb:.2f}MB)")
            else:
                debug_print(f"Segment {idx}: {os.path.basename(path)} (FILE MISSING!)")
        
        # Verify all segments exist
        missing = [p for p in segment_paths if not os.path.exists(p)]
        if missing:
            print(f"ERROR: Missing segment files: {missing}")
            raise ValueError(f"Missing segment files: {missing}")
        
        # Determine output directory
        debug_print("Determining output directory...")
        if output_dir and output_dir.strip():
            debug_print(f"Custom output_dir provided: '{output_dir}'")
            
            # Clean the path and make it relative to ComfyUI output directory
            clean_output_dir = output_dir.lstrip('./\\')
            base_output_dir = folder_paths.get_output_directory()
            final_output_dir = os.path.join(base_output_dir, clean_output_dir)
            final_output_dir = os.path.normpath(final_output_dir)
            
            debug_print(f"Cleaned path: '{clean_output_dir}'")
            debug_print(f"Base output dir: '{base_output_dir}'")
            debug_print(f"Final output dir: '{final_output_dir}'")
            
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = folder_paths.get_output_directory()
            debug_print(f"Using default ComfyUI output directory: '{final_output_dir}'")
        
        # Generate filename with counter (ComfyUI standard)
        debug_print("Generating output filename...")
        counter = 1
        while True:
            filename = f"{output_prefix}_{counter:05d}.{format}"
            final_output_path = os.path.join(final_output_dir, filename)
            if not os.path.exists(final_output_path):
                break
            counter += 1
        
        debug_print(f"Generated filename: '{filename}'")
        debug_print(f"Full output path: '{final_output_path}'")
        
        # Get video info from first segment and verify consistency
        debug_print(f"Analyzing first segment: {segment_paths[0]}")
        cap = cv2.VideoCapture(segment_paths[0])
        if not cap.isOpened():
            print(f"ERROR: Failed to open first video: {segment_paths[0]}")
            raise RuntimeError(f"Failed to open first video: {segment_paths[0]}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        debug_print(f"First segment info: {width}x{height} @ {original_fps:.2f} FPS, {frame_count} frames")
        debug_print(f"Output settings: {width}x{height} @ {fps} FPS")
        debug_print(f"Output: {final_output_path}")
        
        # Verify all segments have same resolution
        debug_print("Verifying segment consistency...")
        for i, seg_path in enumerate(segment_paths):
            debug_print(f"Checking segment {i}: {os.path.basename(seg_path)}")
            cap = cv2.VideoCapture(seg_path)
            if cap.isOpened():
                seg_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                seg_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                seg_fps = cap.get(cv2.CAP_PROP_FPS)
                seg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                debug_print(f"Resolution: {seg_width}x{seg_height}, FPS: {seg_fps:.2f}, Frames: {seg_frames}")
                
                if seg_width != width or seg_height != height:
                    print(f"WARNING: Resolution mismatch! Expected {width}x{height}")
                if abs(seg_fps - original_fps) > 0.1:
                    print(f"WARNING: FPS mismatch! Expected {original_fps:.2f}")
            else:
                print("ERROR: Could not read segment!")
        
        debug_print("Using FFmpeg with H264 for high quality output")
        # Create a temporary file list for FFmpeg concat
        temp_dir = os.path.dirname(final_output_path)
        concat_file = os.path.join(temp_dir, f"concat_{execution_id}.txt")
        
        try:
            # Debug: Check frame counts in each segment
            debug_print("Analyzing segment frame counts before combining...")
            for i, seg_path in enumerate(segment_paths):
                cap = cv2.VideoCapture(seg_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    debug_print(f"Segment {i}: {frame_count} frames")
                else:
                    debug_print(f"Segment {i}: Could not read frame count")
            
            # Create concat file for FFmpeg (direct concatenation without blending)
            with open(concat_file, 'w') as f:
                for seg_path in segment_paths:
                    f.write(f"file '{os.path.abspath(seg_path)}'\n")
            
            debug_print(f"Combining {len(segment_paths)} segments without blending")
            
            # Use FFmpeg to concatenate with H264
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # -y to overwrite
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264",  # H264 codec
                "-preset", "slow",  # Better compression
                "-crf", "18",  # Near-lossless quality
                "-pix_fmt", "yuv420p",  # Compatibility
            ]
            
            # Add metadata handling
            if not save_metadata:
                ffmpeg_cmd.extend(["-map_metadata", "-1"])
            
            ffmpeg_cmd.append(final_output_path)
            
            debug_print(f"Running FFmpeg concat: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
            
            debug_print(f"FFmpeg success: Combined {len(segment_paths)} segments")
            
        finally:
            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)
        
        # Get file size and filename for display
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        display_filename = os.path.basename(final_output_path)
        
        print(f"Combined video saved: {display_filename} ({file_size:.2f}MB)")
        
        # Extract all frames for preview
        debug_print("Extracting all frames for preview...")
        cap = cv2.VideoCapture(final_output_path)
        if not cap.isOpened():
            debug_print("WARNING: Could not open output video for frame extraction")
            # Return empty tensor if we can't read the video
            preview_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            debug_print(f"Total frames in output: {total_frames}")
            
            frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_normalized = frame_rgb.astype(np.float32) / 255.0
                    frames.append(frame_normalized)
                else:
                    debug_print(f"Failed to read frame {i}")
                    break
            
            cap.release()
            
            if frames:
                # Stack frames into tensor
                preview_frames = torch.from_numpy(np.array(frames)).float()
                debug_print(f"Extracted {len(frames)} frames for preview")
            else:
                debug_print("WARNING: No frames could be extracted")
                preview_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        # Return video for display in ComfyUI interface
        preview = {
            "filename": display_filename,
            "subfolder": "",
            "type": "output",
            "format": format
        }
        return {"ui": {"gifs": [preview]}, "result": (final_output_path, preview_frames)}


NODE_CLASS_MAPPINGS = {
    "VideoSegmentCombinerV2": DaxVideoSegmentCombinerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSegmentCombinerV2": "Video Segment Combiner",
}