from ...utils.debug_utils import debug_print

class DaxRuntimeGenerationLengthSet:
    """Set segment length and total length with validation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_length": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "slider"
                }),
                "total_length": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("segment_length", "total_length", "loop_count")
    FUNCTION = "set_generation_length"
    CATEGORY = "Utilities"
    
    def set_generation_length(self, segment_length, total_length):
        # Force total_length to be divisible by segment_length
        if total_length % segment_length != 0:
            # Round UP to next multiple
            loops = (total_length + segment_length - 1) // segment_length
            adjusted_total = loops * segment_length
            debug_print(f"Adjusted {total_length} → {adjusted_total} (divisible by {segment_length})")
        else:
            loops = total_length // segment_length
            adjusted_total = total_length
        
        debug_print(f"segment={segment_length}, total={adjusted_total}, loops={loops}")
        
        return (segment_length, adjusted_total, loops)


# JavaScript extension for dynamic step size
RUNTIME_JS = '''
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Dax.RuntimeGenerationLengthSet",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RuntimeGenerationLengthSet") {
            return;
        }

        const originalNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalNodeCreated?.apply(this, arguments);
            
            const segmentWidget = this.widgets.find(w => w.name === "segment_length");
            const totalWidget = this.widgets.find(w => w.name === "total_length");
            
            if (!segmentWidget || !totalWidget) return;
            
            // Store original callbacks
            const originalSegmentCallback = segmentWidget.callback;
            const originalTotalCallback = totalWidget.callback;
            
            // Store current segment length for total widget callback
            let currentSegmentLength = parseInt(segmentWidget.value) || 1;
            
            // Function to enforce step constraints
            const enforceStep = (value, step) => {
                const remainder = value % step;
                if (remainder === 0) return value;
                // Round to nearest multiple
                return Math.round(value / step) * step;
            };
            
            // Override segment widget callback
            segmentWidget.callback = function(value) {
                // Force segment to be integer
                const segmentLength = Math.round(parseFloat(value)) || 1;
                if (segmentLength !== parseFloat(value)) {
                    // Snap segment widget to integer
                    segmentWidget.value = segmentLength;
                    if (segmentWidget.inputEl) {
                        segmentWidget.inputEl.value = segmentLength;
                    }
                    if (segmentWidget.element) {
                        segmentWidget.element.value = segmentLength;
                    }
                }
                
                console.log("[DaxNodes] Segment changed to:", segmentLength);
                
                currentSegmentLength = segmentLength; // Update stored value
                
                // Update total widget to respect new step
                const currentTotal = totalWidget.value;
                const newTotal = Math.max(enforceStep(currentTotal, segmentLength), segmentLength);
                
                console.log("[DaxNodes] Enforcing total:", currentTotal, "->", newTotal, "step:", segmentLength);
                
                totalWidget.value = newTotal;
                
                // Call original callback with integer value
                if (originalSegmentCallback) {
                    originalSegmentCallback.call(this, segmentLength);
                }
            };
            
            // Override total widget callback to enforce step
            totalWidget.callback = function(value) {
                console.log("[DaxNodes] Total widget changed:", value, "currentSegmentLength:", currentSegmentLength);
                
                const enforcedValue = Math.max(enforceStep(value, currentSegmentLength), currentSegmentLength);
                
                console.log("[DaxNodes] Enforced value:", enforcedValue, "original:", value);
                
                if (enforcedValue !== value) {
                    console.log("[DaxNodes] CORRECTING - changing", value, "to", enforcedValue, "step:", currentSegmentLength);
                    // Set the corrected value without triggering callback again
                    totalWidget.value = enforcedValue;
                    
                    // Update DOM if it exists
                    if (totalWidget.inputEl) {
                        totalWidget.inputEl.value = enforcedValue;
                    }
                    if (totalWidget.element) {
                        totalWidget.element.value = enforcedValue;
                    }
                    
                    value = enforcedValue;
                } else {
                    console.log("[DaxNodes] No correction needed");
                }
                
                // Call original callback with corrected value
                if (originalTotalCallback) {
                    originalTotalCallback.call(this, value);
                }
            };
            
            // Initialize
            segmentWidget.callback(segmentWidget.value);
        };
    }
});
'''

# Create JavaScript file
import os

def write_runtime_js():
    """Write the JavaScript extension for dynamic step size"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.join(current_dir, "..", "..", "web", "comfyui")
    os.makedirs(web_dir, exist_ok=True)
    
    js_file = os.path.join(web_dir, "runtime_generation_length_set.js")
    
    with open(js_file, 'w') as f:
        f.write(RUNTIME_JS)
    
    debug_print(f"Created runtime JS extension: {js_file}")

# Create JS on module load
try:
    write_runtime_js()
except Exception as e:
    print(f"Warning: Could not create runtime JS extension: {e}")


NODE_CLASS_MAPPINGS = {
    "RuntimeGenerationLengthSet": DaxRuntimeGenerationLengthSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RuntimeGenerationLengthSet": "Generation Settings",
}