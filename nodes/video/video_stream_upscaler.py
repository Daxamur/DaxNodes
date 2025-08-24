"""
Video Stream Upscaler Node for ComfyUI

Upscales video files using ComfyUI upscale models.
Memory-efficient stream upscaling for long videos.

Original ComfyUI Copyright (C) 2023 comfyanonymous
ComfyUI is licensed under GPL v3: https://github.com/comfyanonymous/ComfyUI
This implementation: Copyright (C) 2024 DaxNodes
Licensed under GPL v3
"""

import torch
import numpy as np
import os
import cv2
import subprocess
import shutil
import folder_paths
from pathlib import Path
import threading
import queue
import time
import comfy.utils
from comfy import model_management
from ...utils.performance_optimizer import PERF_OPTIMIZER
from ...utils.debug_utils import debug_print

class DaxVideoStreamUpscaler:
    """Stream-upscale video without loading entire video into VRAM"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_filepath": ("STRING", {
                    "tooltip": "Path to input video file"
                }),
                "upscale_model": ("UPSCALE_MODEL",),
                "upscale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "output_prefix": ("STRING", {
                    "default": "upscaled_video",
                    "multiline": False
                }),
                "format": (["mp4", "webm", "mov", "avi"], {"default": "mp4"}),
                "codec": (["h264", "h265", "vp9", "prores"], {"default": "h264"}),
                "buffer_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Batch size for processing (0 = auto-detect based on your system)"
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
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metadata in output video file"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_filepath",)
    FUNCTION = "upscale_video"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    def upscale_video(self, video_filepath, upscale_model, upscale_factor, output_prefix, 
                     format, codec, buffer_size, enable_optimizations=True, output_dir="", save_metadata=True):
        
        if not os.path.exists(video_filepath):
            raise ValueError(f"Input video not found: {video_filepath}")
        
        # Initialize performance optimization
        if enable_optimizations:
            print("Performance-optimized processing enabled")
            debug_print(PERF_OPTIMIZER.get_system_info_summary())
            optimal_settings = PERF_OPTIMIZER.get_optimal_upscaler_settings(upscale_factor)
            
            # Override buffer_size if auto-detect
            if buffer_size == 0:
                buffer_size = optimal_settings["batch_size"]
                debug_print(f"Auto-detected optimal batch size: {buffer_size}")
        else:
            debug_print("Standard processing mode")
            if buffer_size == 0:
                buffer_size = 8  # Default fallback
            optimal_settings = {"batch_size": buffer_size, "tile_size": 512, "overlap": 32}
        
        print(f"Processing video: {os.path.basename(video_filepath)}")
        debug_print(f"Upscale factor: {upscale_factor}x")
        debug_print(f"Batch size: {buffer_size}")
        debug_print(f"Performance profile: {PERF_OPTIMIZER.performance_profile}")
        
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
        
        # Get video info using ffprobe
        video_info = self.get_video_info(video_filepath)
        original_width = video_info['width']
        original_height = video_info['height']
        fps = video_info['fps']
        total_frames = video_info['frames']
        
        # Calculate output dimensions
        target_width = int(original_width * upscale_factor)
        target_height = int(original_height * upscale_factor)
        
        debug_print(f"Input: {original_width}x{original_height} @ {fps:.2f} FPS")
        debug_print(f"Output: {target_width}x{target_height} @ {fps:.2f} FPS")
        debug_print(f"Total frames: {total_frames}")
        
        # Setup temporary directories under .tmp
        base_output_dir = folder_paths.get_output_directory()
        tmp_base_dir = os.path.join(base_output_dir, ".tmp")
        upscaler_temp_dir = os.path.join(tmp_base_dir, "upscaler")
        
        # Create unique temp directories for this run
        import time
        timestamp = str(int(time.time()))
        temp_input_dir = os.path.join(upscaler_temp_dir, f"input_{timestamp}")
        temp_output_dir = os.path.join(upscaler_temp_dir, f"output_{timestamp}")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Use direct video frame extraction for better performance
            if enable_optimizations and optimal_settings.get("use_memory_mapping", True):
                debug_print("Using optimized direct frame processing...")
                self.process_video_optimized(video_filepath, temp_output_dir, upscale_model, 
                                           buffer_size, optimal_settings)
            else:
                debug_print("Using standard frame extraction and processing...")
                # Extract frames from video
                debug_print("Extracting frames...")
                self.extract_frames(video_filepath, temp_input_dir)
                
                # Process frames with intelligent batching
                frame_files = sorted([f for f in os.listdir(temp_input_dir) if f.endswith('.png')])
                debug_print(f"Processing {len(frame_files)} frames with batch size {buffer_size}")
                
                self.process_frames_batched(temp_input_dir, temp_output_dir, frame_files, 
                                          upscale_model, buffer_size, optimal_settings)
            
            # Combine frames back to video
            debug_print("Creating output video...")
            self.combine_frames_to_video(temp_output_dir, final_output_path, fps, codec, save_metadata)
            
        finally:
            # Cleanup temp directories
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            
            # Clean up parent directory if empty
            if os.path.exists(upscaler_temp_dir):
                try:
                    if not os.listdir(upscaler_temp_dir):
                        os.rmdir(upscaler_temp_dir)
                except:
                    pass
        
        # Get final file info
        file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        print(f"Upscaled video saved: {os.path.basename(final_output_path)} ({file_size:.2f}MB)")
        debug_print(f"Full path: {final_output_path}")
        
        return (final_output_path,)
    
    def get_video_info(self, video_path):
        """Get video information using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        import json
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
    
    def extract_frames(self, video_path, output_dir):
        """Extract all frames from video using ffmpeg with lossless PNG"""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-pix_fmt", "rgb24",  # Force RGB color space
            "-compression_level", "0",  # Fastest/lossless PNG compression
            os.path.join(output_dir, "frame_%06d.png")
        ]
        
        debug_print(f"Frame extraction cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            debug_print(f"FFmpeg extraction stderr: {result.stderr}")
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
    
    def upscale_with_async_buffering(self, frame_files, temp_input_dir, temp_output_dir, 
                                   buffer_size, upscale_model, upscale_factor, device):
        """Async double-buffering for 2-3x faster processing"""
        debug_print(f"Using async double-buffering with buffer size: {buffer_size}")
        
        # Create CUDA streams for async processing
        streams = []
        if device.type == 'cuda':
            main_stream = torch.cuda.Stream()
            load_stream = torch.cuda.Stream()
            streams = [main_stream, load_stream]
        
        # Handle ComfyUI model device management BEFORE starting threads
        debug_print(f"Setting up model on device: {device}")
        model_device = device
        actual_model = None
        
        try:
            if hasattr(upscale_model, 'model'):
                debug_print("Found ComfyUI model wrapper")
                # ComfyUI model wrapper - get actual model reference (don't reassign)
                actual_model = upscale_model.model
                # Move the actual model to device
                actual_model.to(device)
                model_device = next(actual_model.parameters()).device
                debug_print(f"ComfyUI model moved to {model_device}")
            else:
                debug_print("Found direct PyTorch model")
                # Direct PyTorch model
                upscale_model = upscale_model.to(device)
                model_device = next(upscale_model.parameters()).device
                debug_print(f"Direct model moved to {model_device}")
                
        except RuntimeError as e:
            if device.type == 'cuda':
                debug_print(f"Failed to move model to CUDA, falling back to CPU: {e}")
                model_device = torch.device('cpu')
                if hasattr(upscale_model, 'model'):
                    actual_model = upscale_model.model
                    actual_model.to(model_device)
                else:
                    upscale_model = upscale_model.to(model_device)
            else:
                raise
        
        # Update device to match actual model device
        debug_print(f"Final device: {model_device}")
        device = model_device
        
        # Setup queues for pipeline parallelism
        load_queue = queue.Queue(maxsize=2)  # Buffer for loaded batches
        error_queue = queue.Queue()
        
        processed_count = 0
        total_batches = (len(frame_files) + buffer_size - 1) // buffer_size
        
        def batch_loader():
            """Background thread that loads batches into pinned memory"""
            try:
                debug_print(f"Starting batch loader for {total_batches} batches...")
                for batch_idx in range(total_batches):
                    batch_start = batch_idx * buffer_size
                    batch_end = min(batch_start + buffer_size, len(frame_files))
                    batch_files = frame_files[batch_start:batch_end]
                    
                    debug_print(f"Loading batch {batch_idx+1}/{total_batches} ({len(batch_files)} frames)...")
                    
                    # Load batch tensors with pinned memory for faster GPU transfer
                    batch_tensors = []
                    output_paths = []
                    
                    for filename in batch_files:
                        input_path = os.path.join(temp_input_dir, filename)
                        output_path = os.path.join(temp_output_dir, filename)
                        output_paths.append(output_path)
                        
                        # Load image
                        image = cv2.imread(input_path)
                        if image is None:
                            raise RuntimeError(f"Could not load image: {input_path}")
                        
                        # Convert BGR to RGB and normalize
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
                        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
                        
                        # Use pinned memory for faster transfer to GPU
                        if device.type == 'cuda':
                            image_tensor = image_tensor.pin_memory()
                        
                        batch_tensors.append(image_tensor)
                    
                    # Stack into batch tensor
                    batch_tensor = torch.stack(batch_tensors)
                    debug_print(f"Batch {batch_idx+1} loaded, shape: {batch_tensor.shape}")
                    
                    # Put loaded batch in queue (blocks if queue is full)
                    debug_print(f"Queuing batch {batch_idx+1}...")
                    load_queue.put((batch_idx, batch_tensor, output_paths))
                    debug_print(f"Batch {batch_idx+1} queued successfully")
                    
                # Signal end of batches
                debug_print("All batches loaded, sending end signal...")
                load_queue.put(None)
                
            except Exception as e:
                error_queue.put(e)
                load_queue.put(None)
        
        # Start background loader thread
        loader_thread = threading.Thread(target=batch_loader)
        loader_thread.start()
        
        try:
            
            # Process batches as they become available
            debug_print("Starting processing loop...")
            while True:
                # Check for loader errors
                if not error_queue.empty():
                    raise error_queue.get()
                
                # Get next loaded batch (blocks until available)
                debug_print("Waiting for next batch...")
                batch_data = load_queue.get()
                if batch_data is None:  # End signal
                    debug_print("Received end signal, finishing...")
                    break
                
                batch_idx, batch_tensor, output_paths = batch_data
                debug_print(f"Processing batch {batch_idx+1} with {len(output_paths)} frames...")
                
                # Transfer to GPU and process with proper stream handling
                if device.type == 'cuda' and len(streams) > 0:
                    # Use CUDA stream for async processing
                    with torch.cuda.stream(streams[0]):
                        batch_tensor = batch_tensor.to(device, non_blocking=True)
                        
                        # Process batch
                        with torch.no_grad():
                            if hasattr(upscale_model, 'upscale'):
                                upscaled_batch = upscale_model.upscale(batch_tensor)
                            else:
                                upscaled_batch = upscale_model(batch_tensor)
                    
                    # CRITICAL: Synchronize AFTER stream context to ensure completion
                    torch.cuda.synchronize()
                    
                else:
                    # CPU or fallback processing
                    batch_tensor = batch_tensor.to(device)
                    with torch.no_grad():
                        if hasattr(upscale_model, 'upscale'):
                            upscaled_batch = upscale_model.upscale(batch_tensor)
                        else:
                            upscaled_batch = upscale_model(batch_tensor)
                
                # Save frames to disk
                for i, output_path in enumerate(output_paths):
                    upscaled_tensor = upscaled_batch[i]
                    upscaled_np = upscaled_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
                    upscaled_np = np.clip(np.round(upscaled_np * 255), 0, 255).astype(np.uint8)
                    upscaled_bgr = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, upscaled_bgr)
                
                processed_count += len(output_paths)
                
                # Progress update
                debug_print(f"Processed batch {batch_idx+1}/{total_batches}: {processed_count}/{len(frame_files)} frames")
                
                # Clear GPU memory periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        finally:
            # Wait for loader thread to finish
            loader_thread.join()
        
        return processed_count
    
    def upscale_single_frame(self, input_path, output_path, upscale_model):
        """Upscale single frame using ComfyUI model pipeline"""
        debug_print(f"UPSCALE_SINGLE_FRAME: Starting for {input_path}")
        from PIL import Image as PILImage
        import numpy as np
        
        # Load through PIL like ComfyUI does
        pil_image = PILImage.open(input_path).convert("RGB")
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # HWC->BHWC
        
        # Debug: Save extracted frame to output for manual testing with native ComfyUI
        if "frame_000001.png" in input_path:
            debug_input_path = input_path.replace("temp_input_frames", "").replace("frame_000001.png", "DEBUG_extracted_frame001.png")
            # Save the PIL image exactly as ComfyUI would see it
            pil_image.save(debug_input_path)
            debug_print(f"DEBUG: Saved extracted frame to {debug_input_path} for manual ComfyUI testing")
        
        debug_print(f"Debug: Input tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        debug_print(f"Debug: Input range: min={image_tensor.min():.3f}, max={image_tensor.max():.3f}")
        
        # Apply upscaling to frame
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image_tensor.movedim(-1, -3).to(device)  # BHWC->BCHW
        
        debug_print(f"Debug: Model input shape: {in_img.shape}, device: {in_img.device}")
        
        # Use ComfyUI's tiled_scale with default settings
        tile = 512
        overlap = 32
        
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
        
        # Convert back exactly like ComfyRoll
        upscaled_tensor = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # BCHW->BHWC
        upscale_model.cpu()
        
        debug_print(f"Debug: Output tensor shape: {upscaled_tensor.shape}, dtype: {upscaled_tensor.dtype}")
        debug_print(f"Debug: Output range: min={upscaled_tensor.min():.3f}, max={upscaled_tensor.max():.3f}")
        
        # Convert to numpy and save
        upscaled_tensor = upscaled_tensor.squeeze(0)  # Remove batch dimension -> HWC
        upscaled_np = upscaled_tensor.cpu().numpy()
        
        # Convert to uint8 with proper rounding
        upscaled_np = np.clip(np.round(upscaled_np * 255), 0, 255).astype(np.uint8)
        
        debug_print(f"Debug: Final RGB shape: {upscaled_np.shape}, dtype: {upscaled_np.dtype}")
        debug_print(f"Debug: RGB range: min={upscaled_np.min()}, max={upscaled_np.max()}")
        
        # Save directly as RGB using PIL instead of BGR conversion
        upscaled_pil = PILImage.fromarray(upscaled_np, mode='RGB')
        upscaled_pil.save(output_path, 'PNG', compress_level=0)
        
        # Debug: Save first frame to output directory for comparison
        if "frame_000001.png" in output_path:
            debug_path = output_path.replace("temp_upscaled_frames", "").replace("frame_000001.png", "DEBUG_upscaled_frame001.png")
            upscaled_pil.save(debug_path, 'PNG', compress_level=0)
            debug_print(f"DEBUG: Saved first upscaled frame to {debug_path} for comparison")
    
    def upscale_frame_batch(self, input_dir, output_dir, batch_files, upscale_model, device):
        """Upscale a batch of frames for better performance"""
        debug_print(f"upscale_frame_batch called with {len(batch_files)} files")
        batch_tensors = []
        output_paths = []
        
        # Move model to correct device (with fallback to CPU)
        debug_print(f"Moving model to device: {device}")
        try:
            if hasattr(upscale_model, 'model'):
                actual_model = upscale_model.model
                actual_model = actual_model.to(device)
                debug_print("Model moved to device (ComfyUI descriptor)")
            else:
                upscale_model = upscale_model.to(device)
                debug_print("Model moved to device (direct model)")
        except RuntimeError as e:
            if device.type == 'cuda':
                debug_print(f"Failed to move model to CUDA, falling back to CPU: {e}")
                device = torch.device('cpu')
                if hasattr(upscale_model, 'model'):
                    actual_model = upscale_model.model
                    actual_model = actual_model.to(device)
                else:
                    upscale_model = upscale_model.to(device)
            else:
                raise
        
        # Load batch of images
        for filename in batch_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            output_paths.append(output_path)
            
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                raise RuntimeError(f"Could not load image: {input_path}")
            
            # Convert BGR to RGB and normalize
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            batch_tensors.append(image_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Process batch
        with torch.no_grad():
            upscaled_batch = upscale_with_model_tiled(batch_tensor, upscale_model, device)
        
        # Save each frame
        for i, output_path in enumerate(output_paths):
            upscaled_tensor = upscaled_batch[i]
            debug_print(f"Saving frame {i}: tensor shape {upscaled_tensor.shape}, dtype {upscaled_tensor.dtype}")
            debug_print(f"Output path: {output_path}")
            
            upscaled_np = upscaled_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            debug_print(f"After permute: shape {upscaled_np.shape}, dtype {upscaled_np.dtype}")
            debug_print(f"Value range: min={upscaled_np.min():.3f}, max={upscaled_np.max():.3f}")
            
            upscaled_np = np.clip(np.round(upscaled_np * 255), 0, 255).astype(np.uint8)
            debug_print(f"After uint8 conversion: shape {upscaled_np.shape}, dtype {upscaled_np.dtype}")
            debug_print(f"Value range: min={upscaled_np.min()}, max={upscaled_np.max()}")
            
            upscaled_bgr = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, upscaled_bgr)
            debug_print(f"cv2.imwrite success: {success}")
            
            # Verify file was written
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                debug_print(f"File written: {file_size} bytes")
            else:
                print("ERROR: File not written!")
                
            # Only print details for first frame to avoid spam
            if i == 0:
                debug_print(f"First frame details: {os.path.basename(output_path)}")

    def combine_frames_to_video(self, frames_dir, output_path, fps, codec, save_metadata=True):
        """Combine frames back to video using ffmpeg"""
        codec_settings = {
            "h264": ["-c:v", "libx264", "-preset", "slow"],
            "h265": ["-c:v", "libx265", "-preset", "slow"],
            "vp9": ["-c:v", "libvpx-vp9", "-speed", "2"],
            "prores": ["-c:v", "prores_ks", "-profile:v", "3"]
        }
        
        # Check if frames exist
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        debug_print(f"Found {len(frame_files)} upscaled frames")
        if len(frame_files) == 0:
            raise RuntimeError(f"No upscaled frames found in {frames_dir}")
        
        # Check first and last frame details
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        last_frame_path = os.path.join(frames_dir, frame_files[-1])
        import cv2
        test_frame = cv2.imread(first_frame_path)
        if test_frame is not None:
            h, w = test_frame.shape[:2]
            debug_print(f"Upscaled frame dimensions: {w}x{h}")
            debug_print(f"First frame: {frame_files[0]}")
            debug_print(f"Last frame: {frame_files[-1]}")
            debug_print(f"Expected pattern: frame_000001.png to frame_{len(frame_files):06d}.png")
            
            # Check if frame naming matches expected pattern
            expected_first = "frame_000001.png"
            expected_last = f"frame_{len(frame_files):06d}.png"
            if frame_files[0] != expected_first:
                print(f"WARNING: Frame naming mismatch! Expected {expected_first}, got {frame_files[0]}")
            if frame_files[-1] != expected_last:
                print(f"WARNING: Frame naming mismatch! Expected {expected_last}, got {frame_files[-1]}")
        else:
            print(f"ERROR: Could not read first frame: {first_frame_path}")
        
        # Use fixed CRF 10 for consistent near-lossless quality
        crf_value = 18  # Visually lossless but reasonable file size
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
        ] + codec_settings.get(codec, codec_settings["h264"]) + [
            "-crf", str(crf_value),
            "-pix_fmt", "yuv420p",  # Changed to yuv420p for Windows compatibility
            "-movflags", "+faststart",  # Better streaming compatibility
        ]
        
        # Add metadata handling
        if not save_metadata:
            cmd.extend(["-map_metadata", "-1"])
        
        cmd.append(output_path)
        
        debug_print(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            debug_print(f"FFmpeg stderr: {result.stderr}")
            debug_print(f"FFmpeg stdout: {result.stdout}")
            raise RuntimeError(f"Video creation failed: {result.stderr}")
        else:
            debug_print("FFmpeg success")
    
    def process_video_optimized(self, video_path, output_dir, upscale_model, batch_size, settings):
        """Optimized direct video processing without intermediate PNG files"""
        import time
        from PIL import Image as PILImage
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        device = model_management.get_torch_device()
        upscale_model.to(device)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        frame_batch = []
        batch_count = 0
        
        # Dynamic tile size based on system capabilities
        tile_size = settings.get("tile_size", 512)
        overlap = settings.get("overlap", 32)
        
        debug_print(f"Direct video processing: {total_frames} frames, tile size: {tile_size}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to RGB tensor
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_batch.append(frame_tensor)
                
                # Process when batch is full or at end of video
                if len(frame_batch) == batch_size or processed_frames + len(frame_batch) == total_frames:
                    # Stack batch for processing
                    if len(frame_batch) == 1:
                        batch_tensor = frame_batch[0].unsqueeze(0)  # Single frame
                    else:
                        batch_tensor = torch.stack(frame_batch)  # Multiple frames
                    
                    # Process batch with adaptive tiling
                    upscaled_batch = self.upscale_batch_optimized(
                        batch_tensor, upscale_model, device, tile_size, overlap
                    )
                    
                    # Save batch results
                    for i, upscaled_frame in enumerate(upscaled_batch):
                        frame_idx = processed_frames + i
                        output_path = os.path.join(output_dir, f"frame_{frame_idx+1:06d}.png")
                        self.save_tensor_as_png(upscaled_frame, output_path)
                    
                    processed_frames += len(frame_batch)
                    frame_batch = []
                    batch_count += 1
                    
                    # Progress update
                    if batch_count % 5 == 0:
                        debug_print(f"Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%)")
                    
                    # Memory management
                    if PERF_OPTIMIZER.should_reduce_batch_size(batch_size):
                        debug_print("Memory pressure detected, clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        finally:
            cap.release()
            upscale_model.cpu()  # Move model back to CPU to free GPU memory
    
    def process_frames_batched(self, input_dir, output_dir, frame_files, upscale_model, batch_size, settings):
        """Process frames in intelligent batches"""
        device = model_management.get_torch_device()
        upscale_model.to(device)
        
        tile_size = settings.get("tile_size", 512)
        overlap = settings.get("overlap", 32)
        
        total_frames = len(frame_files)
        processed_count = 0
        
        try:
            for i in range(0, total_frames, batch_size):
                batch_files = frame_files[i:i+batch_size]
                batch_tensors = []
                
                # Load batch
                for frame_file in batch_files:
                    input_path = os.path.join(input_dir, frame_file)
                    from PIL import Image as PILImage
                    pil_image = PILImage.open(input_path).convert("RGB")
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)
                    batch_tensors.append(image_tensor)
                
                # Stack and process batch
                if len(batch_tensors) == 1:
                    batch_tensor = batch_tensors[0].unsqueeze(0)
                else:
                    batch_tensor = torch.stack(batch_tensors)
                
                # Process with adaptive tiling  
                upscaled_batch = self.upscale_batch_optimized(
                    batch_tensor, upscale_model, device, tile_size, overlap
                )
                
                # Save results
                for j, (frame_file, upscaled_frame) in enumerate(zip(batch_files, upscaled_batch)):
                    output_path = os.path.join(output_dir, frame_file)
                    self.save_tensor_as_png(upscaled_frame, output_path)
                
                processed_count += len(batch_files)
                
                # Progress update
                if (i // batch_size + 1) % 5 == 0:
                    debug_print(f"Processed {processed_count}/{total_frames} frames")
                
                # Adaptive memory management
                if PERF_OPTIMIZER.should_reduce_batch_size(batch_size):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            upscale_model.cpu()
    
    def upscale_batch_optimized(self, batch_tensor, upscale_model, device, tile_size, overlap):
        """Optimized batch upscaling with adaptive tiling"""
        batch_size = batch_tensor.shape[0]
        upscaled_frames = []
        
        for i in range(batch_size):
            frame = batch_tensor[i]
            
            # Convert to ComfyUI format (BCHW)
            in_img = frame.unsqueeze(0).movedim(-1, -3).to(device)
            
            # Adaptive tile size based on image resolution
            current_tile_size = PERF_OPTIMIZER.adaptive_tile_size(
                (frame.shape[0], frame.shape[1]), tile_size
            )
            
            # Use ComfyUI's tiled_scale with OOM handling
            oom = True
            current_tile = current_tile_size
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3], in_img.shape[2], tile_x=current_tile, tile_y=current_tile, overlap=overlap
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    upscaled = comfy.utils.tiled_scale(
                        in_img, lambda a: upscale_model(a), 
                        tile_x=current_tile, tile_y=current_tile, overlap=overlap, 
                        upscale_amount=upscale_model.scale, pbar=pbar
                    )
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    current_tile //= 2
                    if current_tile < 128:
                        print("WARNING: Extremely low memory, using minimal tile size")
                        current_tile = 128
                        upscaled = comfy.utils.tiled_scale(
                            in_img, lambda a: upscale_model(a), 
                            tile_x=current_tile, tile_y=current_tile, overlap=overlap//2, 
                            upscale_amount=upscale_model.scale, pbar=pbar
                        )
                        oom = False
            
            # Convert back to HWC format
            upscaled_frame = torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0).squeeze(0)
            upscaled_frames.append(upscaled_frame.cpu())  # Move to CPU to save GPU memory
        
        return upscaled_frames
    
    def save_tensor_as_png(self, tensor, output_path):
        """Efficiently save tensor as PNG"""
        from PIL import Image as PILImage
        
        # Convert tensor to numpy
        tensor_np = tensor.cpu().numpy()
        
        # Convert to uint8
        image_np = np.clip(np.round(tensor_np * 255), 0, 255).astype(np.uint8)
        
        # Save using PIL for better compression
        pil_image = PILImage.fromarray(image_np, mode='RGB')
        pil_image.save(output_path, 'PNG', compress_level=1)  # Fast compression


NODE_CLASS_MAPPINGS = {
    "VideoStreamUpscaler": DaxVideoStreamUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStreamUpscaler": "Video Upscaler",
}