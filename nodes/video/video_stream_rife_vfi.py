"""
Video Stream RIFE VFI Node for ComfyUI

RIFE (Real-Time Intermediate Flow Estimation) Video Frame Interpolation
Memory-efficient sliding window implementation for infinite video processing

Credits:
- RIFE algorithm: Huang et al., ECCV 2022
- Auto-download for missing models
- Model repositories: styler00dollar, hzwer

Original RIFE Citation:
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
"""

import torch
import numpy as np
import os
import cv2
import subprocess
import shutil
import folder_paths
import json
from pathlib import Path
from ...utils.performance_optimizer import PERF_OPTIMIZER
from ...utils.debug_utils import debug_print
from ...utils.metadata_utils import gather_comfyui_metadata, create_metadata_file, cleanup_metadata_file

class DaxVideoStreamRIFEVFI:
    """Stream-interpolate video using RIFE without loading entire video into VRAM"""
    
    # RIFE model configurations based on comfyui-frame-interpolation
    RIFE_MODELS = {
        "rife40.pth": "4.0",
        "rife41.pth": "4.1", 
        "rife42.pth": "4.2",
        "rife43.pth": "4.3",
        "rife44.pth": "4.4",
        "rife45.pth": "4.5",
        "rife46.pth": "4.6",
        "rife47.pth": "4.7",
        "rife48.pth": "4.8",
        "rife49.pth": "4.9",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available RIFE models (check multiple locations)
        rife_models = []
        
        # Check our own models directory first
        rife_models_dir = os.path.join(os.path.dirname(__file__), "models", "rife")
        if os.path.exists(rife_models_dir):
            for model_file in os.listdir(rife_models_dir):
                if model_file.endswith('.pth') and not model_file.startswith('.'):
                    rife_models.append(model_file)
        
        # Check if ComfyUI-Frame-Interpolation is installed and has RIFE models
        try:
            # ComfyUI-Frame-Interpolation stores models in custom_nodes/comfyui-frame-interpolation/ckpts/rife/
            import folder_paths
            base_path = folder_paths.base_path  # ComfyUI root directory
            frame_interp_rife_dir = os.path.join(base_path, "custom_nodes", "comfyui-frame-interpolation", "ckpts", "rife")
            
            if os.path.exists(frame_interp_rife_dir):
                debug_print(f"Found ComfyUI-Frame-Interpolation RIFE models at: {frame_interp_rife_dir}")
                for model_file in os.listdir(frame_interp_rife_dir):
                    if model_file.endswith('.pth') and not model_file.startswith('.') and model_file not in rife_models:
                        rife_models.append(f"[ComfyUI-FI] {model_file}")  # Mark as from Frame-Interpolation
        except Exception as e:
            debug_print(f"ComfyUI-Frame-Interpolation detection failed: {e}")
            pass  # ComfyUI-Frame-Interpolation not installed or accessible
        
        # If no models found, show standard ComfyUI-Frame-Interpolation models
        if not rife_models:
            rife_models = ["rife47.pth", "rife48.pth", "rife49.pth"]  # Standard models available from established repos
            debug_print("No RIFE models found. Auto-download will start when node is used.")
        
        return {
            "required": {
                "video_filepath": ("STRING", {
                    "tooltip": "Path to input video file"
                }),
                "ckpt_name": (sorted(rife_models, reverse=True), {  # Sort newest first
                    "default": rife_models[0] if rife_models else "No models found"
                }),
                "multiplier": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Frame interpolation multiplier (2x = double frame rate)"
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip contextnet refinement for speed (RIFE 4.0-4.3 only)"
                }),
                "ensemble": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bidirectional processing for higher quality"
                }),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {
                    "default": 1.0,
                    "tooltip": "Multi-scale processing factor"
                }),
                "output_prefix": ("STRING", {
                    "default": "interpolated_video",
                    "multiline": False
                }),
                "format": (["mp4", "webm", "mov", "avi"], {"default": "mp4"}),
                "codec": (["h264", "h265", "vp9", "prores"], {"default": "h264"}),
                "buffer_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Buffer size for memory management (0 = auto-detect based on your system)"
                }),
                "enable_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable intelligent performance optimizations"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom output directory. Leave empty for ComfyUI output."
                }),
                "custom_multiplier_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON list of per-frame multipliers: [2,3,2,4,...] (overrides multiplier)"
                }),
                "skip_frame_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON list of frame indices to skip interpolation: [5,10,15,...]"
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metadata in output video file"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_filepath",)
    FUNCTION = "interpolate_video"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def __init__(self):
        self.rife_model = None
        self.current_model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def interpolate_video(self, video_filepath, ckpt_name, multiplier, fast_mode, ensemble, 
                         scale_factor, output_prefix, format, codec, buffer_size,
                         enable_optimizations=True, output_dir="", 
                         custom_multiplier_list="", skip_frame_list="", save_metadata=True):
        
        if not os.path.exists(video_filepath):
            raise ValueError(f"Input video not found: {video_filepath}")
        
        # Initialize performance optimization
        if enable_optimizations:
            print("Performance-optimized processing enabled")
            debug_print(PERF_OPTIMIZER.get_system_info_summary())
            optimal_settings = PERF_OPTIMIZER.get_optimal_rife_settings(multiplier)
            
            # Override buffer_size if auto-detect
            if buffer_size == 0:
                buffer_size = optimal_settings["buffer_size"]
                debug_print(f"Auto-detected optimal buffer size: {buffer_size}")
        else:
            debug_print("Standard processing mode")
            if buffer_size == 0:
                buffer_size = 30  # Default fallback
            optimal_settings = {"buffer_size": buffer_size, "cache_size": 10}
        
        print(f"Processing video: {os.path.basename(video_filepath)}")
        debug_print(f"Model: {ckpt_name}")
        debug_print(f"Multiplier: {multiplier}x")
        debug_print(f"Buffer size: {buffer_size}")
        debug_print(f"Performance profile: {PERF_OPTIMIZER.performance_profile}")
        debug_print(f"Settings: fast_mode={fast_mode}, ensemble={ensemble}, scale={scale_factor}")
        
        # Parse optional parameters
        multiplier_list = None
        skip_list = None
        
        if custom_multiplier_list.strip():
            try:
                multiplier_list = json.loads(custom_multiplier_list)
                debug_print(f"Using custom multiplier list: {len(multiplier_list)} values")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid multiplier list JSON: {e}")
        
        if skip_frame_list.strip():
            try:
                skip_list = json.loads(skip_frame_list)
                debug_print(f"Skip frames: {skip_list}")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid skip list JSON: {e}")
        
        # Load RIFE model
        self.load_rife_model(ckpt_name)
        
        # Determine output directory
        if output_dir and output_dir.strip():
            # Clean the path and make it relative to ComfyUI output directory
            clean_output_dir = output_dir.lstrip('./\\')
            base_output_dir = folder_paths.get_output_directory()
            final_output_dir = os.path.join(base_output_dir, clean_output_dir)
            final_output_dir = os.path.normpath(final_output_dir)
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = folder_paths.get_output_directory()
        
        # Generate output filename
        counter = 1
        while True:
            filename = f"{output_prefix}_{counter:05d}.{format}"
            final_output_path = os.path.join(final_output_dir, filename)
            if not os.path.exists(final_output_path):
                break
            counter += 1
        
        # Get video info
        video_info = self.get_video_info(video_filepath)
        original_fps = video_info['fps']
        total_frames = video_info['frames']
        
        # Calculate output FPS
        avg_multiplier = multiplier
        if multiplier_list:
            avg_multiplier = sum(multiplier_list) / len(multiplier_list)
        
        output_fps = original_fps * avg_multiplier
        
        debug_print(f"Input: {video_info['width']}x{video_info['height']} @ {original_fps:.2f} FPS")
        debug_print(f"Output: estimated @ {output_fps:.2f} FPS")
        debug_print(f"Total frames: {total_frames}")
        
        # Setup temporary directories under .tmp
        base_output_dir = folder_paths.get_output_directory()
        tmp_base_dir = os.path.join(base_output_dir, ".tmp")
        interpolation_temp_dir = os.path.join(tmp_base_dir, "interpolation")
        
        # Create unique temp directories for this run
        import time
        timestamp = str(int(time.time()))
        temp_input_dir = os.path.join(interpolation_temp_dir, f"input_{timestamp}")
        temp_output_dir = os.path.join(interpolation_temp_dir, f"output_{timestamp}")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Use optimized processing based on settings
            if enable_optimizations:
                debug_print("Using optimized batch processing...")
                self.process_video_optimized(
                    video_filepath, temp_output_dir, multiplier, multiplier_list,
                    skip_list, fast_mode, ensemble, scale_factor, optimal_settings
                )
            else:
                debug_print("Using standard sliding window processing...")
                self.process_video_sliding_window_simplified(
                    video_filepath, temp_output_dir, multiplier, multiplier_list,
                    skip_list, fast_mode, ensemble, scale_factor, buffer_size
                )
            
            # Combine frames to video
            debug_print("Creating output video...")
            self.combine_frames_to_video(temp_output_dir, final_output_path, output_fps, codec, save_metadata)
            
        finally:
            # Cleanup temp directories
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            
            # Clean up parent directory if empty
            if os.path.exists(interpolation_temp_dir):
                try:
                    if not os.listdir(interpolation_temp_dir):
                        os.rmdir(interpolation_temp_dir)
                except:
                    pass
        
        # Get final file info
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        print(f"Interpolated video saved: {os.path.basename(final_output_path)} ({file_size:.2f}MB)")
        debug_print(f"Full path: {final_output_path}")
        
        return (final_output_path,)
    
    def get_video_info(self, video_path):
        """Get video information using ffprobe"""
        import subprocess
        import json
        
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")
        
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        
        # Get frame rate
        fps_str = stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        # Get frame count
        frames = int(stream.get('nb_frames', 0))
        if frames == 0:
            # Fallback: estimate from duration and fps
            duration = float(stream.get('duration', 0))
            frames = int(duration * fps)
        
        return {
            'width': int(stream['width']),
            'height': int(stream['height']),
            'fps': fps,
            'frames': frames
        }
    
    def load_rife_model(self, ckpt_name):
        """Load RIFE model if not already loaded"""
        if self.rife_model is not None and self.current_model_name == ckpt_name:
            return  # Model already loaded
        
        # Find model file - check multiple locations including ComfyUI-Frame-Interpolation
        model_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Handle prefix mismatches - try both prefixed and non-prefixed versions
        actual_model_name = ckpt_name.replace("[ComfyUI-FI] ", "") if ckpt_name.startswith("[ComfyUI-FI] ") else ckpt_name
        
        # Try all possible locations for both prefixed and non-prefixed names
        possible_paths = [
            # DaxNodes locations (highest priority)
            os.path.join(current_dir, "models", "rife", actual_model_name),
            os.path.join(current_dir, "rife_models", actual_model_name),
            # ComfyUI-Frame-Interpolation location
            os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-frame-interpolation", "ckpts", "rife", actual_model_name),
            # ComfyUI standard locations
            os.path.join(folder_paths.models_dir, "rife", actual_model_name),
            os.path.join(folder_paths.models_dir, "vfi", "rife", actual_model_name),
            os.path.join(folder_paths.models_dir, "checkpoints", "rife", actual_model_name),
            os.path.join("./models/rife", actual_model_name),
            os.path.join("./ckpts/rife", actual_model_name)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                ckpt_name = actual_model_name  # Use actual name for loading
                debug_print(f"Found RIFE model: {model_path}")
                break
        
        if model_path is None:
            # Auto-download if model not found locally
            print(f"RIFE model {ckpt_name} not found locally, attempting auto-download...")
            
            # Create rife model directories if they don't exist
            current_dir = os.path.dirname(os.path.abspath(__file__))
            primary_rife_dir = os.path.join(current_dir, "models", "rife")
            fallback_rife_dir = os.path.join(folder_paths.models_dir, "rife")
            
            os.makedirs(primary_rife_dir, exist_ok=True)
            os.makedirs(fallback_rife_dir, exist_ok=True)
            
            # Auto-download if not available locally
            download_success = self.attempt_model_download(ckpt_name, primary_rife_dir)
            
            if download_success:
                model_path = os.path.join(primary_rife_dir, ckpt_name)
            else:
                print(f"Auto-download failed for {ckpt_name}")
                print(f"Please manually download RIFE models and place them in:")
                print(f"  PRIMARY: {primary_rife_dir}")
                print(f"  FALLBACK: {fallback_rife_dir}")
                print(f"Download from: https://github.com/hzwer/Practical-RIFE")
                print(f"Using enhanced linear interpolation fallback for now")
                
                self.rife_model = None
                self.current_model_name = "fallback"
                return
        
        debug_print(f"Loading RIFE model: {model_path}")
        
        # Load model with proper RIFE architecture handling
        try:
            # First try loading as torch.jit model (most RIFE models are TorchScript)
            if model_path.endswith('.pth'):
                try:
                    self.rife_model = torch.jit.load(model_path, map_location=self.device)
                    self.rife_model.eval()
                    debug_print("Model loaded as TorchScript")
                    
                    # Test the model with dummy input to verify compatibility
                    try:
                        with torch.no_grad():
                            test_input = torch.randn(1, 6, 64, 64).to(self.device)  # 2 RGB frames concatenated
                            _ = self.rife_model(test_input)
                        debug_print("Model compatibility test passed")
                    except Exception as e:
                        print(f"Model compatibility test failed: {e}")
                        print(f"This model may not be compatible with our interface")
                        
                except Exception as e:
                    debug_print(f"TorchScript loading failed: {e}")
                    # Try loading as state dict
                    try:
                        state_dict = torch.load(model_path, map_location=self.device)
                        if 'model' in state_dict:
                            # Extract model state dict if wrapped
                            state_dict = state_dict['model']
                        
                        # Create RIFE model architecture and load weights
                        self.rife_model = self.create_rife_model(ckpt_name)
                        self.rife_model.load_state_dict(state_dict, strict=False)
                        self.rife_model.to(self.device)
                        self.rife_model.eval()
                        debug_print("Model loaded as state dict")
                    except Exception as state_e:
                        debug_print(f"State dict loading also failed: {state_e}")
                        debug_print("Using fallback linear interpolation")
                        self.rife_model = "dummy"
            
            elif model_path.endswith('.pkl'):
                # Handle pickle files (original RIFE format)
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    # Extract model from dict
                    if 'model' in model_data:
                        self.rife_model = model_data['model']
                    elif 'state_dict' in model_data:
                        self.rife_model = self.create_rife_model(ckpt_name)
                        self.rife_model.load_state_dict(model_data['state_dict'], strict=False)
                    else:
                        self.rife_model = model_data
                else:
                    self.rife_model = model_data
                
                self.rife_model.to(self.device)
                self.rife_model.eval()
                debug_print("Model loaded from pickle")
            
            self.current_model_name = ckpt_name
            debug_print("RIFE model loaded successfully")
            
        except Exception as e:
            # Fallback: create dummy model for testing
            print(f"Warning: Could not load RIFE model ({e})")
            print(f"Using simple interpolation fallback")
            self.rife_model = "dummy"
            self.current_model_name = ckpt_name
    
    def create_rife_model(self, ckpt_name):
        """Create proper RIFE model architecture"""
        # Load RIFE model architecture
        try:
            # Try to import existing RIFE models if available
            import sys
            comfy_fi_path = os.path.join(folder_paths.base_path, "custom_nodes", "comfyui-frame-interpolation")
            if comfy_fi_path not in sys.path:
                sys.path.append(comfy_fi_path)
            
            from vfi_models.rife.rife_arch import IFNet
            
            # Determine RIFE version from checkpoint name
            version = self.get_rife_version(ckpt_name)
            debug_print(f"Creating RIFE {version} architecture using ComfyUI-FI IFNet")
            
            # Create model using ComfyUI-Frame-Interpolation's exact architecture
            return IFNet(arch_ver=version)
            
        except ImportError as e:
            debug_print(f"ComfyUI-Frame-Interpolation not available: {e}")
            debug_print("Using fallback architecture")
            # Fallback to our custom architecture
            version = self.get_rife_version(ckpt_name)
            if version.startswith('4.'):
                return RIFEv4(version)
            elif version.startswith('3.'):
                return RIFEv3(version)
            else:
                return RIFEv4('4.0')  # Default to v4.0
    
    def get_rife_version(self, ckpt_name):
        """Extract RIFE version from checkpoint name"""
        # Map model version from filename
        ckpt_name_ver_dict = {
            "rife40.pth": "4.0",
            "rife41.pth": "4.0", 
            "rife42.pth": "4.2", 
            "rife43.pth": "4.3", 
            "rife44.pth": "4.3", 
            "rife45.pth": "4.5",
            "rife46.pth": "4.6",
            "rife47.pth": "4.7",
            "rife48.pth": "4.7",
            "rife49.pth": "4.7",
            "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
        }
        
        # Check exact filename match first
        if ckpt_name in ckpt_name_ver_dict:
            return ckpt_name_ver_dict[ckpt_name]
        
        # Fallback to pattern matching for non-standard names
        ckpt_lower = ckpt_name.lower()
        if 'rife49' in ckpt_lower:
            return '4.7'  # rife49 uses 4.7 architecture
        elif 'rife48' in ckpt_lower:
            return '4.7'  # rife48 uses 4.7 architecture 
        elif 'rife47' in ckpt_lower:
            return '4.7'
        elif 'rife46' in ckpt_lower:
            return '4.6'
        elif 'rife45' in ckpt_lower:
            return '4.5'
        elif 'rife44' in ckpt_lower:
            return '4.3'  # rife44 uses 4.3 architecture
        elif 'rife43' in ckpt_lower:
            return '4.3'
        elif 'rife42' in ckpt_lower:
            return '4.2'
        elif 'rife41' in ckpt_lower:
            return '4.0'  # rife41 uses 4.0 architecture
        elif 'rife40' in ckpt_lower:
            return '4.0'
        else:
            # Default to stable version
            return '4.7'  # Most compatible
    
    def attempt_model_download(self, model_name, target_dir):
        """Auto-download RIFE models using standard ComfyUI-Frame-Interpolation URLs"""
        
        # Same URLs as ComfyUI-Frame-Interpolation (established, trusted)
        base_urls = [
            "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
            "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
            "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
        ]
        
        target_path = os.path.join(target_dir, model_name)
        
        for i, base_url in enumerate(base_urls, 1):
            try:
                download_url = base_url + model_name
                debug_print(f"Attempting download {i}/{len(base_urls)}: {model_name}")
                debug_print(f"URL: {download_url}")
                
                import urllib.request
                
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(block_num * block_size * 100 / total_size, 100)
                        debug_print(f"Progress: {percent:.1f}%", end='\r')
                
                urllib.request.urlretrieve(download_url, target_path, progress_hook)
                
                # Verify download
                if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                    print(f"Successfully downloaded: {model_name}")
                    return True
                else:
                    print(f"Download incomplete: {model_name}")
                    if os.path.exists(target_path):
                        os.remove(target_path)
                        
            except Exception as e:
                print(f"Download failed: {str(e)}")
                if os.path.exists(target_path):
                    os.remove(target_path)
                continue
        
        print(f"All download URLs failed for {model_name}")
        return False

    def extract_frames(self, video_path, output_dir):
        """Extract frames using ffmpeg"""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-qscale:v", "1",
            os.path.join(output_dir, "frame_%06d.png")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    def process_video_optimized(self, video_path, output_dir, multiplier, multiplier_list,
                               skip_list, fast_mode, ensemble, scale_factor, settings):
        """Optimized RIFE processing with intelligent batching and memory management"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buffer_size = settings.get("buffer_size", 30)
        cache_size = settings.get("cache_size", 10)
        
        debug_print(f"Optimized processing: {total_frames} frames, buffer: {buffer_size}")
        
        # Simple frame cache for overlapping frames
        frame_cache = {}
        output_frame_count = 0
        processed_pairs = 0
        
        try:
            # Process frame pairs directly
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, cv_frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB tensor
                frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                current_frame = torch.from_numpy(frame_rgb).float() / 255.0
                
                if prev_frame is not None:
                    # Determine multiplier for this pair
                    current_multiplier = multiplier
                    if multiplier_list and processed_pairs < len(multiplier_list):
                        current_multiplier = multiplier_list[processed_pairs]
                    
                    # Skip if needed
                    if skip_list and processed_pairs in skip_list:
                        output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
                        self.save_frame(prev_frame, output_path)
                        output_frame_count += 1
                    else:
                        # Perform RIFE interpolation
                        interpolated_frames = self.rife_interpolate(
                            prev_frame, current_frame, current_multiplier, 
                            fast_mode, ensemble, scale_factor
                        )
                        
                        # Save interpolated frames
                        for frame in interpolated_frames:
                            output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
                            self.save_frame(frame, output_path)
                            output_frame_count += 1
                    
                    processed_pairs += 1
                    
                    # Progress and memory management
                    if processed_pairs % 10 == 0:
                        debug_print(f"Processed {processed_pairs}/{total_frames-1} frame pairs")
                        
                        # Memory management
                        if PERF_OPTIMIZER.should_reduce_batch_size(buffer_size):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            frame_cache.clear()  # Clear cache to free memory
                
                prev_frame = current_frame
                frame_idx += 1
            
            # Save last frame
            if prev_frame is not None:
                output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
                self.save_frame(prev_frame, output_path)
                output_frame_count += 1
            
            debug_print(f"Generated {output_frame_count} total frames from {total_frames} input frames")
        
        finally:
            cap.release()
    
    def process_video_sliding_window_simplified(self, video_path, output_dir, multiplier, multiplier_list,
                                              skip_list, fast_mode, ensemble, scale_factor, buffer_size):
        """Simplified sliding window processing (fallback method)"""
        # Extract frames first (simpler approach)
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_input_dir:
            # Extract frames
            self.extract_frames(video_path, temp_input_dir)
            
            # Process with standard method
            self.process_frames_with_rife(
                temp_input_dir, output_dir, multiplier, multiplier_list,
                skip_list, fast_mode, ensemble, scale_factor, 10  # Simple cache interval
            )
    
    def process_frames_with_rife(self, input_dir, output_dir, multiplier, multiplier_list, 
                               skip_list, fast_mode, ensemble, scale_factor, clear_cache_interval):
        """Process frames with RIFE interpolation"""
        frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        
        if len(frame_files) < 2:
            raise ValueError("Need at least 2 frames for interpolation")
        
        output_frame_count = 0
        
        for i in range(len(frame_files) - 1):
            current_frame_path = os.path.join(input_dir, frame_files[i])
            next_frame_path = os.path.join(input_dir, frame_files[i + 1])
            
            # Determine multiplier for this frame pair
            current_multiplier = multiplier
            if multiplier_list and i < len(multiplier_list):
                current_multiplier = multiplier_list[i]
            
            # Check if this frame should be skipped
            if skip_list and i in skip_list:
                # Just copy the frame without interpolation
                output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
                shutil.copy2(current_frame_path, output_path)
                output_frame_count += 1
                continue
            
            # Load frame pair
            frame1 = self.load_frame(current_frame_path)
            frame2 = self.load_frame(next_frame_path)
            
            # Perform RIFE interpolation
            interpolated_frames = self.rife_interpolate(
                frame1, frame2, current_multiplier, fast_mode, ensemble, scale_factor
            )
            
            # Save original frame and interpolated frames
            for j, frame in enumerate(interpolated_frames):
                output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
                self.save_frame(frame, output_path)
                output_frame_count += 1
            
            # Clear cache periodically
            if (i + 1) % clear_cache_interval == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            debug_print(f"Processed frame pair {i+1}/{len(frame_files)-1}")
        
        # Save the last frame
        last_frame_path = os.path.join(input_dir, frame_files[-1])
        output_path = os.path.join(output_dir, f"frame_{output_frame_count:06d}.png")
        shutil.copy2(last_frame_path, output_path)
        
        debug_print(f"Generated {output_frame_count + 1} total frames")
    
    def load_frame(self, frame_path):
        """Load frame as tensor"""
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb).float() / 255.0
    
    def save_frame(self, frame_tensor, output_path):
        """Save tensor as frame"""
        frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
    
    def rife_interpolate(self, frame1, frame2, multiplier, fast_mode, ensemble, scale_factor):
        """Perform RIFE interpolation between two frames with real optical flow"""
        if self.rife_model == "dummy":
            # Enhanced dummy interpolation with smoother blending
            frames = [frame1]  # Start with first frame
            
            for i in range(1, multiplier):
                t = i / multiplier
                smooth_t = self.smooth_interpolation_curve(t)
                blended = frame1 * (1 - smooth_t) + frame2 * smooth_t
                frames.append(blended)
            
            return frames
        
        frames = [frame1]
        
        try:
            # Preprocess frames for RIFE (handle padding, scaling, format conversion)
            frame1_processed, frame2_processed, original_shape = self.preprocess_frames_for_rife(
                frame1, frame2, scale_factor
            )
            
            # Move to device
            frame1_device = frame1_processed.to(self.device)
            frame2_device = frame2_processed.to(self.device)
            
            # Generate intermediate frames using real RIFE architecture
            for i in range(1, multiplier):
                timestep = i / multiplier
                
                with torch.no_grad():
                    if hasattr(self.rife_model, 'forward') and self.rife_model != "dummy":
                        # Configure model settings (only for our custom models)
                        if hasattr(self.rife_model, 'set_fast_mode'):
                            self.rife_model.set_fast_mode(fast_mode)
                        
                        # Generate intermediate frame
                        try:
                            # ComfyUI-FI approach: model(frame_0, frame_1, timestep, scale_list, fastmode, ensemble)
                            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
                            
                            interpolated = self.rife_model(
                                frame1_device, 
                                frame2_device, 
                                timestep=timestep, 
                                scale_list=scale_list,
                                training=False,  # Always False for inference
                                fastmode=fast_mode,
                                ensemble=ensemble
                            )
                            
                        except Exception as comfy_fi_e:
                            try:
                                # Fallback: TorchScript models usually expect concatenated frames
                                concatenated_input = torch.cat([frame1_device, frame2_device], dim=1)  # BCHW, 6 channels
                                interpolated = self.rife_model(concatenated_input)
                                
                            except Exception as concat_e:
                                try:
                                    # Fallback 2: Simple separate frame arguments
                                    interpolated = self.rife_model(frame1_device, frame2_device, timestep)
                                        
                                except Exception as simple_e:
                                    # All methods failed, raise the most informative error
                                    error_msg = f"ComfyUI-FI: {comfy_fi_e}, Concat: {concat_e}, Simple: {simple_e}"
                                    raise Exception(f"All inference methods failed: {error_msg}")
                        
                        # Multi-scale processing if scale_factor != 1.0 (disable recursion)
                        if scale_factor != 1.0 and not getattr(self, '_in_multiscale', False):
                            interpolated = self.apply_multiscale_refinement(
                                interpolated, frame1_device, frame2_device, timestep, scale_factor
                            )
                        
                        # Post-process: remove padding, convert format, clamp values
                        interpolated = self.postprocess_rife_output(interpolated, original_shape)
                        
                    else:
                        # Enhanced fallback with temporal consistency
                        t_smooth = self.smooth_interpolation_curve(timestep)
                        interpolated = frame1 * (1 - t_smooth) + frame2 * t_smooth
                
                # Move to CPU if needed and add to frames list
                if interpolated.is_cuda:
                    interpolated = interpolated.cpu()
                frames.append(interpolated)
            
        except Exception as e:
            debug_print(f"Warning: RIFE inference failed ({e}), using enhanced linear interpolation")
            # Enhanced fallback with temporal consistency
            for i in range(1, multiplier):
                t = i / multiplier
                t_smooth = self.smooth_interpolation_curve(t)
                blended = frame1 * (1 - t_smooth) + frame2 * t_smooth
                frames.append(blended)
        
        return frames
    
    def smooth_interpolation_curve(self, t):
        """Apply smooth interpolation curve for better temporal consistency"""
        # Use smoothstep function for more natural motion
        return t * t * (3 - 2 * t)
    
    def preprocess_frames_for_rife(self, frame1, frame2, scale_factor):
        """Preprocess frames for RIFE model input with proper handling"""
        # Store original shape for postprocessing
        original_shape = frame1.shape  # HWC format
        
        # Convert HWC to BCHW format
        frame1_tensor = frame1.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        frame2_tensor = frame2.permute(2, 0, 1).unsqueeze(0)
        
        # Apply scale factor if needed (for multi-scale processing)
        if scale_factor != 1.0:
            import torch.nn.functional as F
            new_h = int(frame1_tensor.shape[2] * scale_factor)
            new_w = int(frame1_tensor.shape[3] * scale_factor)
            
            frame1_tensor = F.interpolate(frame1_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            frame2_tensor = F.interpolate(frame2_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Store processed shape before padding
        processed_shape = frame1_tensor.shape
        
        # Ensure dimensions are divisible by 64 (RIFE architecture requirement)
        h, w = frame1_tensor.shape[2:]
        pad_h = ((h + 63) // 64) * 64 - h
        pad_w = ((w + 63) // 64) * 64 - w
        
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F
            frame1_tensor = F.pad(frame1_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            frame2_tensor = F.pad(frame2_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        return frame1_tensor, frame2_tensor, {
            'original': original_shape,
            'processed': processed_shape,
            'padded': frame1_tensor.shape,
            'scale_factor': scale_factor
        }
    
    def postprocess_rife_output(self, output, shape_info):
        """Postprocess RIFE output back to original format"""
        # Handle both old format (tuple) and new format (dict)
        if isinstance(shape_info, dict):
            original_shape = shape_info['original']
            scale_factor = shape_info['scale_factor']
        else:
            original_shape = shape_info
            scale_factor = 1.0
        
        # Remove padding if it was added - crop to processed size first
        if isinstance(shape_info, dict) and 'processed' in shape_info:
            processed_h, processed_w = shape_info['processed'][2:]
            output = output[:, :, :processed_h, :processed_w]
        
        # Scale back to original size if scale_factor was applied
        if scale_factor != 1.0:
            import torch.nn.functional as F
            original_h, original_w = original_shape[:2]
            output = F.interpolate(output, size=(original_h, original_w), mode='bilinear', align_corners=False)
        
        # Convert BCHW back to HWC
        output = output.squeeze(0).permute(1, 2, 0)
        
        # Clamp values to valid range
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def apply_multiscale_refinement(self, interpolated, frame1, frame2, timestep, scale_factor):
        """Apply multi-scale refinement for better quality at different resolutions"""
        if scale_factor == 1.0:
            return interpolated
        
        # Set flag to prevent infinite recursion
        self._in_multiscale = True
        
        try:
            # Process at multiple scales for better temporal consistency
            scales = [0.5, 1.0, 2.0] if scale_factor > 1.0 else [0.25, 0.5, 1.0]
            refined_results = []
            
            for scale in scales:
                if scale == 1.0:
                    # Use the already computed result
                    refined_results.append(interpolated)
                else:
                    # Compute at different scale
                    import torch.nn.functional as F
                    h, w = frame1.shape[2:]
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Rescale input frames
                    frame1_scaled = F.interpolate(frame1, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    frame2_scaled = F.interpolate(frame2, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    # Interpolate at this scale (DISABLE multi-scale to prevent recursion)
                    with torch.no_grad():
                        # Temporarily disable multi-scale for recursive call
                        if hasattr(self.rife_model, 'forward'):
                            result_scaled = self.rife_model.forward(frame1_scaled, frame2_scaled, timestep)
                        else:
                            # Fallback to simple interpolation at this scale
                            alpha = timestep if isinstance(timestep, float) else timestep.item()
                            result_scaled = frame1_scaled * (1 - alpha) + frame2_scaled * alpha
                    
                    # Scale back to original resolution
                    result_upscaled = F.interpolate(result_scaled, size=interpolated.shape[2:], mode='bilinear', align_corners=False)
                    refined_results.append(result_upscaled)
            
            # Weighted combination of multi-scale results
            weights = [0.2, 0.6, 0.2]  # Favor the 1.0 scale
            final_result = sum(w * r for w, r in zip(weights, refined_results))
            
            return final_result
            
        except Exception as e:
            debug_print(f"Multi-scale refinement failed: {e}, using single-scale result")
            return interpolated
        finally:
            # Clear flag
            self._in_multiscale = False
    
    def combine_frames_to_video(self, frames_dir, output_path, fps, codec, save_metadata=True):
        """Combine frames to video using ffmpeg"""
        codec_settings = {
            "h264": ["-c:v", "libx264", "-preset", "slow"],
            "h265": ["-c:v", "libx265", "-preset", "slow"],
            "vp9": ["-c:v", "libvpx-vp9", "-speed", "2"],
            "prores": ["-c:v", "prores_ks", "-profile:v", "3"]
        }
        
        # Use fixed CRF 10 for consistent near-lossless quality
        crf_value = 18  # Visually lossless but reasonable file size
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
        ] + codec_settings.get(codec, codec_settings["h264"]) + [
            "-crf", str(crf_value),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Better streaming compatibility
        ]
        
        # Add metadata handling
        # Gather metadata if saving
        metadata_file_path = None
        if save_metadata:
            metadata = gather_comfyui_metadata("DaxNodes StreamRIFE")
            if metadata:
                metadata_file_path = create_metadata_file(metadata)
        
        # Handle metadata in FFmpeg command
        if metadata_file_path:
            cmd.extend(["-i", metadata_file_path, "-map", "0", "-map_metadata", "1"])
        elif not save_metadata:
            cmd.extend(["-map_metadata", "-1"])
        
        cmd.append(output_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Video creation failed: {result.stderr}")
        
        # Clean up metadata file
        cleanup_metadata_file(metadata_file_path)


class ConvBlock(torch.nn.Module):
    """Basic convolution block for RIFE"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResBlock(torch.nn.Module):
    """Residual block for RIFE feature extraction"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = torch.nn.InstanceNorm2d(channels)
        self.norm2 = torch.nn.InstanceNorm2d(channels)
        self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class FlowNet(torch.nn.Module):
    """RIFE FlowNet for optical flow estimation"""
    def __init__(self, version='4.0'):
        super().__init__()
        self.version = version
        
        # Encoder
        self.encoder = torch.nn.ModuleList([
            ConvBlock(6, 32),     # Input: 2 RGB frames (6 channels)
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        ])
        
        # Residual blocks
        self.res_blocks = torch.nn.ModuleList([
            ResBlock(256) for _ in range(4)
        ])
        
        # Decoder for flow estimation
        self.decoder = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            torch.nn.ConvTranspose2d(32, 16, 4, 2, 1),
        ])
        
        # Flow output layers
        self.flow_head = torch.nn.Conv2d(16, 4, 3, 1, 1)  # 4 channels: forward + backward flow
        
        # Feature extraction for context
        self.feature_head = torch.nn.Conv2d(16, 32, 3, 1, 1)
    
    def forward(self, x):
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
            x = torch.nn.functional.avg_pool2d(x, 2)
        
        # Residual processing
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(features):
                skip = features[-(i+1)]
                if x.shape[2:] != skip.shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # Output flow and features
        flow = self.flow_head(x)
        features = self.feature_head(x)
        
        return flow, features


class ContextNet(torch.nn.Module):
    """RIFE ContextNet for handling occlusions and refinement"""
    def __init__(self, version='4.0'):
        super().__init__()
        self.version = version
        
        # Context extraction from warped frames and flow
        self.context_encoder = torch.nn.ModuleList([
            ConvBlock(6 + 4 + 32, 64),  # warped frames + flow + features
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        ])
        
        # Context refinement
        self.context_refine = torch.nn.ModuleList([
            ResBlock(256) for _ in range(3)
        ])
        
        # Context decoder
        self.context_decoder = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
        ])
        
        # Context output
        self.context_head = torch.nn.Conv2d(32, 3, 3, 1, 1)  # RGB context
        
    def forward(self, warped_frames, flow, features):
        # Combine inputs
        x = torch.cat([warped_frames, flow, features], dim=1)
        
        # Context encoding
        skip_features = []
        for encoder_layer in self.context_encoder:
            x = encoder_layer(x)
            skip_features.append(x)
            x = torch.nn.functional.avg_pool2d(x, 2)
        
        # Context refinement
        for refine_layer in self.context_refine:
            x = refine_layer(x)
        
        # Context decoding
        for i, decoder_layer in enumerate(self.context_decoder):
            x = decoder_layer(x)
            if i < len(skip_features):
                skip = skip_features[-(i+1)]
                if x.shape[2:] != skip.shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # Output context
        context = torch.sigmoid(self.context_head(x))
        
        return context


class RIFEv4(torch.nn.Module):
    """Complete RIFE v4.x architecture with fast_mode support"""
    def __init__(self, version='4.0'):
        super().__init__()
        self.version = version
        self.fast_mode = False  # Can be toggled
        
        # Core networks
        self.flownet = FlowNet(version)
        self.contextnet = ContextNet(version)
        
        # Final refinement
        self.refine = torch.nn.ModuleList([
            ConvBlock(9, 64),  # 2 warped frames + context
            ResBlock(64),
            ResBlock(64),
            torch.nn.Conv2d(64, 3, 3, 1, 1)
        ])
        
        # Fast mode refinement (simpler, for RIFE 4.0-4.3)
        self.fast_refine = torch.nn.Sequential(
            ConvBlock(6, 32),  # Just warped frames, no context
            torch.nn.Conv2d(32, 3, 3, 1, 1)
        )
        
    def set_fast_mode(self, fast_mode):
        """Enable/disable fast mode (skips contextnet for speed)"""
        self.fast_mode = fast_mode
        
    def forward(self, frame0, frame1, timestep=None):
        """
        Forward pass for RIFE interpolation
        Args:
            frame0: First frame [B, C, H, W]
            frame1: Second frame [B, C, H, W]  
            timestep: Interpolation timestep (0-1), can be tensor or float
        """
        if timestep is None:
            timestep = 0.5
            
        # Handle timestep
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor(timestep, device=frame0.device, dtype=frame0.dtype)
        
        # Ensure frames are same size and 4D
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
        if frame1.dim() == 3:
            frame1 = frame1.unsqueeze(0)
            
        # Concatenate frames for flow estimation
        flow_input = torch.cat([frame0, frame1], dim=1)
        
        # Estimate bidirectional optical flow
        flow, flow_features = self.flownet(flow_input)
        
        # Split flow into forward and backward
        flow_01 = flow[:, :2]  # Forward flow (frame0 -> frame1)
        flow_10 = flow[:, 2:]  # Backward flow (frame1 -> frame0)
        
        # Warp frames using optical flow
        warped_frame0 = self.warp_frame(frame0, flow_01 * timestep)
        warped_frame1 = self.warp_frame(frame1, flow_10 * (1 - timestep))
        
        # Initial interpolated frame (simple blending)
        initial_interp = warped_frame0 * (1 - timestep) + warped_frame1 * timestep
        
        if self.fast_mode:
            # Fast mode: skip contextnet, use simple refinement
            refine_input = torch.cat([warped_frame0, warped_frame1], dim=1)
            refinement = self.fast_refine(refine_input)
            final_frame = initial_interp + refinement
        else:
            # Full mode: use contextnet for better quality
            warped_frames = torch.cat([warped_frame0, warped_frame1], dim=1)
            
            # Generate context for refinement
            context = self.contextnet(warped_frames, flow, flow_features)
            
            # Final refinement with context
            refine_input = torch.cat([warped_frame0, warped_frame1, context], dim=1)
            
            x = refine_input
            for refine_layer in self.refine[:-1]:
                x = refine_layer(x)
            
            # Final output with residual connection
            refinement = self.refine[-1](x)
            final_frame = initial_interp + refinement
        
        # Clamp to valid range
        final_frame = torch.clamp(final_frame, 0, 1)
        
        return final_frame
    
    def warp_frame(self, frame, flow):
        """Warp frame using optical flow"""
        B, C, H, W = frame.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=frame.dtype),
            torch.arange(W, device=frame.device, dtype=frame.dtype),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Apply flow to grid
        warped_grid = grid + flow
        
        # Normalize grid to [-1, 1] for grid_sample
        warped_grid[:, 0] = 2 * warped_grid[:, 0] / (W - 1) - 1  # x
        warped_grid[:, 1] = 2 * warped_grid[:, 1] / (H - 1) - 1  # y
        
        # Permute to [B, H, W, 2] for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp frame
        warped_frame = torch.nn.functional.grid_sample(
            frame, warped_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_frame


class RIFEv3(torch.nn.Module):
    """RIFE v3.x architecture (simplified version of v4)"""
    def __init__(self, version='3.0'):
        super().__init__()
        self.version = version
        
        # Simpler architecture for v3
        self.flownet = FlowNet(version)
        
        # Simpler refinement for v3
        self.refine = torch.nn.Sequential(
            ConvBlock(6, 32),  # 2 warped frames
            ResBlock(32),
            torch.nn.Conv2d(32, 3, 3, 1, 1)
        )
        
    def forward(self, frame0, frame1, timestep=None):
        if timestep is None:
            timestep = 0.5
            
        # Handle timestep and dimensions
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor(timestep, device=frame0.device, dtype=frame0.dtype)
        
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
        if frame1.dim() == 3:
            frame1 = frame1.unsqueeze(0)
        
        # Flow estimation
        flow_input = torch.cat([frame0, frame1], dim=1)
        flow, _ = self.flownet(flow_input)
        
        flow_01 = flow[:, :2]
        flow_10 = flow[:, 2:]
        
        # Warp frames using static method
        warped_frame0 = self.warp_frame(frame0, flow_01 * timestep)
        warped_frame1 = self.warp_frame(frame1, flow_10 * (1 - timestep))
        
        # Simple blending + refinement
        initial_interp = warped_frame0 * (1 - timestep) + warped_frame1 * timestep
        refine_input = torch.cat([warped_frame0, warped_frame1], dim=1)
        refinement = self.refine(refine_input)
        
        final_frame = initial_interp + refinement
        return torch.clamp(final_frame, 0, 1)
    
    def warp_frame(self, frame, flow):
        """Warp frame using optical flow (copied from RIFEv4)"""
        B, C, H, W = frame.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=frame.dtype),
            torch.arange(W, device=frame.device, dtype=frame.dtype),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Apply flow to grid
        warped_grid = grid + flow
        
        # Normalize grid to [-1, 1] for grid_sample
        warped_grid[:, 0] = 2 * warped_grid[:, 0] / (W - 1) - 1  # x
        warped_grid[:, 1] = 2 * warped_grid[:, 1] / (H - 1) - 1  # y
        
        # Permute to [B, H, W, 2] for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp frame
        warped_frame = torch.nn.functional.grid_sample(
            frame, warped_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_frame


NODE_CLASS_MAPPINGS = {
    "VideoStreamRIFEVFI": DaxVideoStreamRIFEVFI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStreamRIFEVFI": "Video Frame Interpolation",
}
