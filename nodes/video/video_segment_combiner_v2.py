import torch
import numpy as np
import os
import folder_paths
import time
import cv2
import json
import subprocess
from ...utils.metadata_utils import gather_comfyui_metadata, create_metadata_file, cleanup_metadata_file

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
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_filepath", "preview_frames")
    FUNCTION = "combine_segments_v2"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def load_segments_from_directory(self, execution_id):
        """Scan execution directory and find all video segments"""
        try:
            base_output_dir = folder_paths.get_output_directory()
            tmp_base_dir = os.path.join(base_output_dir, ".tmp")
            execution_dir = os.path.join(tmp_base_dir, str(execution_id))
            
            if not os.path.exists(execution_dir):
                return {}
            
            segments = {}
            files = os.listdir(execution_dir)
            
            for file in files:
                if file.endswith(('.mp4', '.webm', '.mov', '.avi')):
                    try:
                        name_part = file.split('.')[0]
                        parts = name_part.split('_')
                        
                        if len(parts) >= 2:
                            file_exec_id = parts[0]
                            loop_index_str = parts[1]
                            
                            if file_exec_id == str(execution_id):
                                try:
                                    loop_index = int(loop_index_str)
                                    file_path = os.path.join(execution_dir, file)
                                    segments[loop_index] = file_path
                                except ValueError:
                                    pass
                    except Exception:
                        pass
            
            return segments
        except Exception:
            return {}
    
    def extract_frames_from_video(self, video_path, num_frames=1, from_end=False):
        """Extract specific frames from a video file using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            if from_end:
                start_frame = max(0, total_frames - num_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(min(num_frames, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
            
            return frames if len(frames) > 1 else (frames[0] if frames else None)
        finally:
            cap.release()
    
    
    def combine_segments_v2(self, execution_id, output_prefix, fps, format, codec, output_dir="", save_metadata=True, prompt=None, extra_pnginfo=None):
        
        print(f"Combining segments for execution {execution_id}")
        
        segments = self.load_segments_from_directory(execution_id)
        
        if segments is None:
            print(f"ERROR: load_segments_from_directory returned None for execution {execution_id}")
            raise ValueError(f"Failed to load segments for execution_id={execution_id}")
        
        if not segments:
            print(f"ERROR: No segments found for execution {execution_id}")
            raise ValueError(f"No video segments found in execution directory for execution_id={execution_id}")
        
        sorted_indices = sorted(segments.keys())
        segment_paths = [segments[idx] for idx in sorted_indices]
        
        print(f"Found {len(segment_paths)} segments to combine")
        for idx, path in enumerate(segment_paths):
            if not os.path.exists(path):
                print(f"ERROR: Missing segment file: {os.path.basename(path)}")
        
        missing = [p for p in segment_paths if not os.path.exists(p)]
        if missing:
            print(f"ERROR: Missing segment files: {missing}")
            raise ValueError(f"Missing segment files: {missing}")
        
        if output_dir and output_dir.strip():
            clean_output_dir = output_dir.lstrip('./\\')
            base_output_dir = folder_paths.get_output_directory()
            final_output_dir = os.path.join(base_output_dir, clean_output_dir)
            final_output_dir = os.path.normpath(final_output_dir)
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = folder_paths.get_output_directory()
        counter = 1
        while True:
            filename = f"{output_prefix}_{counter:05d}.{format}"
            final_output_path = os.path.join(final_output_dir, filename)
            if not os.path.exists(final_output_path):
                break
            counter += 1
        
        
        # Get video info from first segment and verify consistency
        cap = cv2.VideoCapture(segment_paths[0])
        if not cap.isOpened():
            print(f"ERROR: Failed to open first video: {segment_paths[0]}")
            raise RuntimeError(f"Failed to open first video: {segment_paths[0]}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        
        # Verify all segments have same resolution
        for i, seg_path in enumerate(segment_paths):
            cap = cv2.VideoCapture(seg_path)
            if cap.isOpened():
                seg_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                seg_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                seg_fps = cap.get(cv2.CAP_PROP_FPS)
                seg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                
                if seg_width != width or seg_height != height:
                    print(f"WARNING: Resolution mismatch! Expected {width}x{height}")
                if abs(seg_fps - original_fps) > 0.1:
                    print(f"WARNING: FPS mismatch! Expected {original_fps:.2f}")
            else:
                print("ERROR: Could not read segment!")
        
        # Create a temporary file list for FFmpeg concat
        temp_dir = os.path.dirname(final_output_path)
        concat_file = os.path.join(temp_dir, f"concat_{execution_id}.txt")
        
        try:
            # Create concat file for FFmpeg
            with open(concat_file, 'w') as f:
                for seg_path in segment_paths:
                    f.write(f"file '{os.path.abspath(seg_path)}'\n")
            
            
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
            
            # Gather metadata if saving
            metadata_file_path = None
            if save_metadata:
                print(f"Gathering metadata with prompt={prompt is not None}, extra_pnginfo={extra_pnginfo is not None}")
                video_metadata = gather_comfyui_metadata("DaxNodes SegmentCombiner", prompt, extra_pnginfo)
                print(f"Video metadata result: {video_metadata is not None}")
                if video_metadata:
                    metadata_file_path = create_metadata_file(video_metadata)
                    print(f"Metadata file created: {metadata_file_path is not None}")
                else:
                    print("No metadata generated!")
            
            # For concat, we need a different approach - concat doesn't support multiple inputs
            if metadata_file_path:
                # Generate temp filename with same extension
                base_path, ext = os.path.splitext(final_output_path)
                temp_output = f"{base_path}_temp{ext}"
                
                try:
                    # First pass: concat without metadata
                    ffmpeg_cmd.append(temp_output)
                    
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
                    
                    # Second pass: add metadata (metadata file first, then video)
                    metadata_cmd = ["ffmpeg", "-y", "-i", metadata_file_path, "-i", temp_output,
                                   "-c", "copy", "-map", "1:v", "-map", "1:a?",
                                   "-metadata", "creation_time=now", final_output_path]
                    result = subprocess.run(metadata_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg metadata pass failed: {result.stderr}")
                        
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                
                # Continue to preview frame extraction - don't return early
            else:
                # Only run if we didn't already process with metadata
                if not save_metadata:
                    ffmpeg_cmd.extend(["-map_metadata", "-1"])
                ffmpeg_cmd.append(final_output_path)
            
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
                
            
        finally:
            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)
            
            # Clean up metadata file
            cleanup_metadata_file(metadata_file_path)
        
        # Get file size and filename for display
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        display_filename = os.path.basename(final_output_path)
        
        print(f"Combined video saved: {display_filename} ({file_size:.2f}MB)")
        
        # Extract all frames for preview
        cap = cv2.VideoCapture(final_output_path)
        if not cap.isOpened():
            # Return empty tensor if we can't read the video
            preview_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_normalized = frame_rgb.astype(np.float32) / 255.0
                    frames.append(frame_normalized)
                else:
                    break
            
            cap.release()
            
            if frames:
                # Stack frames into tensor
                preview_frames = torch.from_numpy(np.array(frames)).float()
            else:
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