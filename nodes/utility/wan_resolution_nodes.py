import torch
import torch.nn.functional as F
import math
from ...utils.debug_utils import debug_print

class DaxWANResolutionPicker:
    """Unified WAN resolution picker with I2V toggle"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "i2v_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle between T2V (False) and I2V (True) modes"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            },
            "optional": {
                # I2V inputs
                "image": ("IMAGE",),
                "i2v_resolution_mode": ([
                    "Native (Original Resolution)",
                    "High (1280x720 Pixel Count)", 
                    "Low (480x854 Pixel Count)"
                ], {
                    "tooltip": "I2V scaling mode (only used when I2V mode enabled)"
                }),
                # T2V inputs  
                "t2v_resolution_preset": ([
                    "720x1280 (9:16 Portrait)",
                    "1280x720 (16:9 Landscape)", 
                    "1024x576 (16:9 Medium)",
                    "576x1024 (9:16 Medium)",
                    "768x768 (1:1 Square)",
                    "640x640 (1:1 Square Medium)",
                    "512x512 (1:1 Square Small)",
                    "480x854 (9:16 Low)",
                    "854x480 (16:9 Low)",
                    "Custom (use override)"
                ], {
                    "tooltip": "T2V resolution presets (only used when I2V mode disabled)"
                }),
                # Override inputs - available in both modes
                "override_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override width (0 = use preset/calculated)"
                }),
                "override_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override height (0 = use preset/calculated)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("mode", "width", "height", "image")
    FUNCTION = "pick_resolution"
    CATEGORY = "Utilities"
    
    def pick_resolution(self, i2v_mode, batch_size, image=None, i2v_resolution_mode="Native (Original Resolution)", 
                       t2v_resolution_preset="1280x720 (16:9 Landscape)", override_width=0, override_height=0):
        
        if i2v_mode:
            # I2V Mode - process input image
            if image is None:
                raise ValueError("Image input required when I2V mode is enabled")
            
            # Get original image dimensions
            batch_size_img, orig_height, orig_width, channels = image.shape
            orig_aspect = orig_width / orig_height
            
            debug_print(f"I2V: Original image - {orig_width}x{orig_height} (aspect {orig_aspect:.3f})")
            
            # Use override if specified
            if override_width > 0 and override_height > 0:
                target_width, target_height = override_width, override_height
                debug_print(f"I2V: Using override - {target_width}x{target_height}")
            else:
                if i2v_resolution_mode == "Native (Original Resolution)":
                    target_width, target_height = orig_width, orig_height
                    debug_print("I2V: Using native resolution")
                    
                elif i2v_resolution_mode == "High (1280x720 Pixel Count)":
                    # Scale to match 1280x720 = 921,600 pixels while maintaining aspect ratio
                    target_pixels = 1280 * 720  # 921,600 pixels
                    scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                    target_width = int(orig_width * scale_factor)
                    target_height = int(orig_height * scale_factor)
                    debug_print("I2V: Scaling to high resolution (~921k pixels)")
                    
                elif i2v_resolution_mode == "Low (480x854 Pixel Count)":
                    # Scale to match 480x854 = 409,920 pixels while maintaining aspect ratio  
                    target_pixels = 480 * 854  # 409,920 pixels
                    scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                    target_width = int(orig_width * scale_factor)
                    target_height = int(orig_height * scale_factor)
                    debug_print("I2V: Scaling to low resolution (~410k pixels)")
            
            # Ensure dimensions are multiples of 8 for VAE compatibility
            target_width = (target_width // 8) * 8
            target_height = (target_height // 8) * 8
            
            # Resize image if needed
            if target_width != orig_width or target_height != orig_height:
                debug_print(f"I2V: Resizing from {orig_width}x{orig_height} to {target_width}x{target_height}")
                
                # Resize using torch interpolate
                image_resized = F.interpolate(
                    image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # BCHW -> BHWC
                
                output_image = image_resized
            else:
                output_image = image
            
            actual_pixels = target_width * target_height
            print(f"I2V resolution: {target_width}x{target_height} ({actual_pixels:,} pixels)")
            
            return (2, target_width, target_height, output_image)
        
        else:
            # T2V Mode - use resolution presets
            debug_print("T2V: Using presets")
            
            # Resolution presets optimized for WAN 2.2
            presets = {
                "720x1280 (9:16 Portrait)": (720, 1280),
                "1280x720 (16:9 Landscape)": (1280, 720),
                "1024x576 (16:9 Medium)": (1024, 576),
                "576x1024 (9:16 Medium)": (576, 1024),
                "768x768 (1:1 Square)": (768, 768),
                "640x640 (1:1 Square Medium)": (640, 640),
                "512x512 (1:1 Square Small)": (512, 512),
                "480x854 (9:16 Low)": (480, 854),
                "854x480 (16:9 Low)": (854, 480),
                "Custom (use override)": (512, 512)  # fallback
            }
            
            # Use override if specified, otherwise use preset
            if override_width > 0 and override_height > 0:
                width, height = override_width, override_height
                debug_print(f"T2V: Using override - {width}x{height}")
            else:
                width, height = presets[t2v_resolution_preset]
                debug_print(f"T2V: Using preset {t2v_resolution_preset} - {width}x{height}")
            
            # Ensure dimensions are multiples of 8 for VAE compatibility
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            print(f"T2V resolution: {width}x{height}")
            
            # Create empty image tensor for consistency
            empty_image = torch.zeros([batch_size, height, width, 3])
            
            return (1, width, height, empty_image)


NODE_CLASS_MAPPINGS = {
    "WANResolutionPicker": DaxWANResolutionPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANResolutionPicker": "Resolution Picker",
}