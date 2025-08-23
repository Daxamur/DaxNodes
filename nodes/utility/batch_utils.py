from ...utils.debug_utils import debug_print

class DaxTrimBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "frames_from_end": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim_batch"
    CATEGORY = "Utilities"

    def trim_batch(self, images, frames_from_end=1):
        debug_print(f"TRIM BATCH DEBUG: Starting trim_batch function")
        debug_print(f"  frames_from_end input: {frames_from_end} (type: {type(frames_from_end)})")
        
        # Handle None or invalid inputs
        if frames_from_end is None:
            print("WARNING: frames_from_end is None, defaulting to 0 (no trim)")
            frames_from_end = 0
        
        # Debug input images
        if images is None:
            print("ERROR: images input is None!")
            import torch
            return (torch.zeros(1, 64, 64, 3),)
        
        debug_print(f"images input: {type(images)}")
        
        # Check if images is a tensor
        if not hasattr(images, 'shape'):
            print(f"ERROR: images has no 'shape' attribute! Type: {type(images)}")
            debug_print(f"images content: {images}")
            # Try to handle string inputs that might be passed incorrectly
            if isinstance(images, str):
                print("ERROR: Received string instead of IMAGE tensor!")
                debug_print(f"String content: '{images}'")
                import torch
                return (torch.zeros(1, 64, 64, 3),)
            else:
                raise TypeError(f"Expected IMAGE tensor, got {type(images)}")
        
        total_frames = images.shape[0]
        debug_print(f"Total frames in batch: {total_frames}")
        debug_print(f"Image tensor shape: {images.shape}")
        debug_print(f"Image tensor dtype: {images.dtype}")
        
        if frames_from_end <= 0:
            debug_print("No trimming requested (frames_from_end <= 0)")
            return (images,)
        
        if frames_from_end >= total_frames:
            print(f"WARNING: Trimming {frames_from_end} >= total frames {total_frames}, returning empty tensor")
            import torch
            return (torch.zeros(0, images.shape[1], images.shape[2], images.shape[3]),)
        
        # Trim from the end - keep everything up to (total - frames_from_end)  
        trim_point = total_frames - frames_from_end
        debug_print(f"Trim point: {trim_point} (keeping frames 0 to {trim_point-1})")
        
        try:
            trimmed_batch = images[:trim_point]
            debug_print(f"Trimmed batch shape: {trimmed_batch.shape}")
            debug_print(f"SUCCESS: Original {total_frames} frames -> Trimmed to {trim_point} frames (removed last {frames_from_end})")
            return (trimmed_batch,)
        except Exception as e:
            debug_print(f"ERROR during slicing: {e}")
            debug_print(f"Attempted slice: images[:{trim_point}]")
            debug_print(f"Images shape: {images.shape}")
            raise


NODE_CLASS_MAPPINGS = {
    "TrimBatch": DaxTrimBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimBatch": "Batch Trimmer",
}