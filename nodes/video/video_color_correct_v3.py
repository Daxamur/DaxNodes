"""
Video Color Correction Node for ComfyUI

Color matching using color-matcher library with support for multiple algorithms.
Color correction algorithms for video consistency.

Portions of this code use the color-matcher library:
- Library: https://github.com/hahnec/color-matcher
- License: GNU General Public License v3.0
- Usage: Color transfer algorithms (hm, mkl, mvgd, reinhard, etc.)

Copyright (c) 2025 DaxNodes
Licensed under MIT License
"""

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
from ...utils.debug_utils import debug_print

# Import color-matcher library
try:
    from color_matcher import ColorMatcher
    HAS_COLOR_MATCHER = True
    debug_print("color-matcher library found - color matching available")
except ImportError:
    HAS_COLOR_MATCHER = False
    debug_print("color-matcher library not found - install with: pip install color-matcher")

class DaxVideoColorCorrect:
    """Color correction node for video consistency"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "method": ([
                    "hm-mvgd-hm", 
                    "hm-mkl-hm", 
                    "mkl", 
                    "mvgd", 
                    "hm", 
                    "reinhard"
                ], {
                    "default": "hm-mvgd-hm",
                    "tooltip": "Color matching method (compound methods work best for video)"
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Correction strength"
                }),
            },
            "optional": {
                "anchor_frame": ("IMAGE", {"tooltip": "Pristine reference frame for long video consistency"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"
    CATEGORY = "Video"
    
    
    def __init__(self):
        self.color_matcher = None
        if HAS_COLOR_MATCHER:
            self.color_matcher = ColorMatcher()
    
    def detect_processed_content(self, images_np, anchor_frame):
        """Detect if content has been over-processed using anchor frame comparison"""
        if anchor_frame is None:
            return False, 1.0
        
        anchor_np = anchor_frame.cpu().numpy()[0]
        
        # Compare color distribution characteristics
        test_frame = images_np[0]
        
        # Calculate various metrics
        anchor_std = np.std(anchor_np)
        test_std = np.std(test_frame)
        
        # Color range comparison
        anchor_range = np.max(anchor_np) - np.min(anchor_np)
        test_range = np.max(test_frame) - np.min(test_frame)
        
        # Histogram similarity
        anchor_hist = np.histogram(anchor_np.flatten(), bins=50, range=(0, 1))[0]
        test_hist = np.histogram(test_frame.flatten(), bins=50, range=(0, 1))[0]
        hist_correlation = np.corrcoef(anchor_hist, test_hist)[0, 1]
        
        # Determine if content appears processed
        std_ratio = test_std / (anchor_std + 1e-8)
        range_ratio = test_range / (anchor_range + 1e-8)
        
        is_processed = (
            std_ratio < 0.7 or std_ratio > 1.4 or  # Extreme std changes
            range_ratio < 0.8 or  # Reduced dynamic range
            hist_correlation < 0.7  # Different histogram shape
        )
        
        # Calculate adaptive strength
        if is_processed:
            # More processed = gentler correction
            processing_severity = max(
                abs(1.0 - std_ratio),
                abs(1.0 - range_ratio),
                1.0 - hist_correlation if hist_correlation > 0 else 0.5
            )
            adaptive_strength = max(0.2, 1.0 - processing_severity)
        else:
            adaptive_strength = 1.0
        
        debug_print(f"Content analysis: processed={is_processed}, adaptive_strength={adaptive_strength:.2f}")
        debug_print(f"  std_ratio={std_ratio:.3f}, range_ratio={range_ratio:.3f}, hist_corr={hist_correlation:.3f}")
        
        return is_processed, adaptive_strength
    
    def color_match_single(self, source, target, method, strength):
        """Color matching implementation for single frame"""
        if not HAS_COLOR_MATCHER:
            # Fallback to simple mean matching if color-matcher not available
            source_mean = np.mean(source, axis=(0, 1))
            target_mean = np.mean(target, axis=(0, 1))
            offset = target_mean - source_mean
            result = source + offset * strength
            return np.clip(result, 0, 1)
        
        try:
            # Convert to uint8 for color-matcher
            source_uint8 = (source * 255).astype(np.uint8)
            target_uint8 = (target * 255).astype(np.uint8)
            
            # Apply color matching using selected method
            matched_uint8 = self.color_matcher.transfer(
                src=source_uint8,
                ref=target_uint8,
                method=method
            )
            
            # Convert back to float
            matched_float = matched_uint8.astype(np.float32) / 255.0
            
            # Apply strength blending
            result = source + strength * (matched_float - source)
            
            return np.clip(result, 0, 1)
            
        except Exception as e:
            debug_print(f"Color matching failed: {e}, falling back to mean matching")
            # Fallback to mean matching
            source_mean = np.mean(source, axis=(0, 1))
            target_mean = np.mean(target, axis=(0, 1))
            offset = target_mean - source_mean
            result = source + offset * strength
            return np.clip(result, 0, 1)
    
    def process_frame_batch(self, frames, target_frame, method, strength):
        """Process multiple frames sequentially to avoid threading issues"""
        results = []
        for frame in frames:
            result = self.color_match_single(frame, target_frame, method, strength)
            results.append(result)
        return results

    def color_correct(self, images, reference_video, method="hm-mvgd-hm", strength=0.8, anchor_frame=None):
        # If anchor frame is connected, this is multi-segment workflow - disable correction
        if anchor_frame is not None:
            debug_print("V3 PASS-THROUGH: Anchor frame detected - skipping to prevent progressive corruption")
            return (images,)
        
        images_np = images.cpu().numpy().copy()
        ref_np = reference_video.cpu().numpy()
        
        if ref_np.shape[0] == 0:
            return (images,)
        
        debug_print(f"COLOR CORRECT V3: method='{method}' for {images_np.shape[0]} frames (single-clip only)")
        
        # Use reference video last frame as target
        target_frame = ref_np[-1]
        debug_print("Using reference video last frame as target")
        
        # Use fixed strength without complex processing detection
        effective_strength = strength
        debug_print(f"Using strength: {effective_strength:.2f}")
        
        # Process frames
        debug_print(f"Applying {method} color matching...")
        
        # Process all frames sequentially
        results = []
        for i, frame in enumerate(images_np):
            result = self.kjnodes_color_match_single(frame, target_frame, method, effective_strength)
            results.append(result)
            
            if i < 3:
                orig_mean = np.mean(frame, axis=(0,1))
                result_mean = np.mean(result, axis=(0,1))
                debug_print(f"Frame {i}: ({orig_mean[0]:.3f}, {orig_mean[1]:.3f}, {orig_mean[2]:.3f}) → ({result_mean[0]:.3f}, {result_mean[1]:.3f}, {result_mean[2]:.3f})")
        
        # Verify results
        result_array = np.array(results)
        input_mean = np.mean(images_np)
        output_mean = np.mean(result_array)
        
        debug_print(f"Processing complete: input_mean={input_mean:.3f}, output_mean={output_mean:.3f}")
        
        return (torch.from_numpy(result_array).float(),)


NODE_CLASS_MAPPINGS = {
    "VideoColorCorrectV3": DaxVideoColorCorrect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoColorCorrectV3": "Video Color Correction",
}