import { app } from "../../scripts/app.js";

/*
 * Video preview functionality adapted from ComfyUI-VideoHelperSuite
 * Original source: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
 * License: GNU General Public License v3.0
 * 
 * Portions of this code are based on or adapted from ComfyUI-VideoHelperSuite,
 * specifically the video widget implementation patterns, event handling, and
 * sizing logic. We comply with GPL v3.0 by:
 * - Maintaining this attribution notice
 * - Our project is also GPL v3.0 compatible
 * - Source code is available
 */

// Add video display support for our video nodes
app.registerExtension({
    name: "Dax.VideoDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only VideoPreview should have video display
        if (nodeData.name === "VideoPreview") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalNodeCreated?.apply(this, arguments);
                
                // Store original onExecuted for this specific node instance
                const originalOnExecuted = this.onExecuted;
                
                // Create unique onExecuted handler for this node instance
                this.onExecuted = function(message) {
                    console.log(`[DaxNodes] ========== NODE EXECUTED ==========`);
                    console.log(`[DaxNodes] Node: ${this.title} (${this.id})`);
                    console.log(`[DaxNodes] Message received:`, message);
                    
                    if (originalOnExecuted) {
                        originalOnExecuted.apply(this, arguments);
                    }
                    
                    if (message?.gifs && message.gifs.length > 0) {
                        console.log(`[DaxNodes] Found gifs in message:`, message.gifs);
                        console.log(`[DaxNodes] Using first gif:`, message.gifs[0]);
                        // Only update if we don't already have a video for this exact file
                        const currentGif = message.gifs[0];
                        if (!this.currentVideoFile || this.currentVideoFile !== currentGif.filename) {
                            this.currentVideoFile = currentGif.filename;
                            this.updateVideoPreview(currentGif);
                        }
                    } else {
                        console.log(`[DaxNodes] NO GIFS FOUND IN MESSAGE!`);
                        console.log(`[DaxNodes] Message keys:`, Object.keys(message || {}));
                    }
                    
                    console.log(`[DaxNodes] ========== NODE EXECUTION END ==========`);
                };
                
                // Restore original sizing with centering - fit largest dimension to node width
                const fitVideoToNode = (videoElement, containerElement) => {
                    if (videoElement && containerElement && videoElement.videoWidth && videoElement.videoHeight) {
                        const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
                        const nodeWidth = containerElement.clientWidth || 300;
                        
                        // Calculate size maintaining aspect ratio, fit to node width
                        let displayWidth = nodeWidth;
                        let displayHeight = nodeWidth / aspectRatio;
                        
                        // Set video size
                        videoElement.style.width = `${displayWidth}px`;
                        videoElement.style.height = `${displayHeight}px`;
                        videoElement.style.display = 'block';
                        videoElement.style.margin = '0 auto'; // Center horizontally
                        
                        // Center container vertically
                        containerElement.style.display = 'flex';
                        containerElement.style.alignItems = 'center';
                        containerElement.style.justifyContent = 'center';
                        containerElement.style.height = `${displayHeight}px`;
                    }
                };
                
                // Create video preview widget (VHS implementation)
                const nodeId = this.id || `node_${Math.random().toString(36).substr(2, 9)}`;
                
                // Clean up existing video widgets
                if (this.widgets) {
                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        const widget = this.widgets[i];
                        if (widget.name?.startsWith("video_preview_")) {
                            if (widget.element) {
                                widget.element.remove();
                            }
                            this.removeWidget(widget);
                        }
                    }
                }
                
                // Main container element (VHS pattern)
                const element = document.createElement("div");
                const previewNode = this;
                
                // Create widget with exact VHS parameters
                const previewWidget = this.addDOMWidget(`video_preview_${nodeId}`, "preview", element, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() {
                        return element.value;
                    },
                    setValue(v) {
                        element.value = v;
                    },
                });
                
                // Dynamic computeSize based on video aspect ratio - prevent auto resize
                previewWidget.computeSize = function(width) {
                    if (previewWidget.aspectRatio) {
                        const videoHeight = width / previewWidget.aspectRatio;
                        return [width, videoHeight];
                    }
                    return [width, 200]; // Default height before video loads
                };
                
                // Event delegation (VHS pattern)
                element.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    return app.canvas._mousedown_callback(e);
                }, true);
                element.addEventListener('pointerdown', (e) => {
                    e.preventDefault();
                    return app.canvas._mousedown_callback(e);
                }, true);
                element.addEventListener('mousewheel', (e) => {
                    e.preventDefault();
                    return app.canvas._mousewheel_callback(e);
                }, true);
                element.addEventListener('pointermove', (e) => {
                    e.preventDefault();
                    return app.canvas._mousemove_callback(e);
                }, true);
                element.addEventListener('pointerup', (e) => {
                    e.preventDefault();
                    return app.canvas._mouseup_callback(e);
                }, true);
                
                // Widget value setup
                previewWidget.value = {hidden: false, paused: false, params: {}};
                
                // Parent container (VHS pattern)
                previewWidget.parentEl = document.createElement("div");
                previewWidget.parentEl.className = "dax_preview";
                previewWidget.parentEl.style['width'] = "100%";
                element.appendChild(previewWidget.parentEl);
                
                // Video element (VHS pattern)
                previewWidget.videoEl = document.createElement("video");
                previewWidget.videoEl.controls = false;
                previewWidget.videoEl.loop = true;
                previewWidget.videoEl.muted = true;
                previewWidget.videoEl.style['width'] = "100%";
                
                // Video event handlers - center video without resizing node
                previewWidget.videoEl.addEventListener("loadedmetadata", () => {
                    previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
                    fitVideoToNode(previewWidget.videoEl, previewWidget.parentEl);
                });
                previewWidget.videoEl.addEventListener("error", () => {
                    previewWidget.parentEl.hidden = true;
                });
                
                previewWidget.parentEl.appendChild(previewWidget.videoEl);
                
                // Update video source
                this.updateVideoPreview = function(params) {
                    if (params.filename && params.type === "output") {
                        let videoUrl;
                        if (params.subfolder && params.subfolder !== "") {
                            videoUrl = `/view?filename=${encodeURIComponent(params.filename)}&subfolder=${encodeURIComponent(params.subfolder)}&type=${params.type}`;
                        } else {
                            videoUrl = `/view?filename=${encodeURIComponent(params.filename)}&type=${params.type}`;
                        }
                        previewWidget.videoEl.src = videoUrl;
                        previewWidget.videoEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;
                    }
                    previewWidget.value.params = params;
                };
            };
        }
    }
});