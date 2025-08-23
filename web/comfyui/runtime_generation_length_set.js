
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
            const overrideWidget = this.widgets.find(w => w.name === "total_length_override");
            
            if (!segmentWidget || !totalWidget || !overrideWidget) return;
            
            // Store original callbacks
            const originalSegmentCallback = segmentWidget.callback;
            const originalTotalCallback = totalWidget.callback;
            const originalOverrideCallback = overrideWidget.callback;
            
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
                
                // Update override widget to respect new step
                const currentOverride = overrideWidget.value;
                if (currentOverride > 0) {
                    const newOverride = enforceStep(currentOverride, segmentLength);
                    console.log("[DaxNodes] Enforcing override:", currentOverride, "->", newOverride, "step:", segmentLength);
                    overrideWidget.value = newOverride;
                }
                
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
            
            // Override total widget callback to enforce step
            overrideWidget.callback = function(value) {
                console.log("[DaxNodes] Override widget changed:", value, "currentSegmentLength:", currentSegmentLength);
                
                if (value > 0) {
                    const enforcedValue = enforceStep(value, currentSegmentLength);
                    
                    console.log("[DaxNodes] Enforced override value:", enforcedValue, "original:", value);
                    
                    if (enforcedValue !== value) {
                        console.log("[DaxNodes] CORRECTING override - changing", value, "to", enforcedValue, "step:", currentSegmentLength);
                        overrideWidget.value = enforcedValue;
                        
                        if (overrideWidget.inputEl) {
                            overrideWidget.inputEl.value = enforcedValue;
                        }
                        if (overrideWidget.element) {
                            overrideWidget.element.value = enforcedValue;
                        }
                        
                        value = enforcedValue;
                    }
                }
                
                // Call original callback
                if (originalOverrideCallback) {
                    originalOverrideCallback.call(this, value);
                }
            };
            
            // Initialize
            segmentWidget.callback(segmentWidget.value);
        };
    }
});
