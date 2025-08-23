# DaxNodes - Professional ComfyUI Node Pack

A comprehensive collection of high-performance video processing and utility nodes for ComfyUI, designed for professional workflows and optimized for memory efficiency.

## Features

- **Advanced Video Processing**: Color correction, frame interpolation, upscaling, and segmentation
- **Smart Face Detection**: MediaPipe-powered face analysis with eye state detection
- **Memory Optimized**: Stream processing for handling large videos without VRAM limits
- **Production Ready**: Professional error handling, debug controls, and clean interfaces
- **Metadata Support**: Optional metadata preservation in all video outputs

## Installation

### Automatic Installation (Recommended)

1. Clone into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Daxamur/DaxNodes.git
```

2. Install FFmpeg on your system:
```bash
# Windows (using chocolatey)
choco install ffmpeg

# macOS (using homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

OR https://www.ffmpeg.org/download.html

3. Python dependencies will auto-install on first load. If needed, manually install:
```bash
cd DaxNodes
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 
- **FFmpeg** (system installation required for video processing)
- OpenCV (`cv2`)
- MediaPipe (for face detection)
- SciPy (for color correction)
- NumPy
- color-matcher (for advanced color matching algorithms)

## Nodes Overview

### Video Processing

#### **Video Color Correction**
Advanced color matching using multiple algorithms (hm-mvgd-hm, hm-mkl-hm, mkl, mvgd, hm, reinhard).
- **Input**: Images, Reference Video, Method, Strength, Optional Anchor Frame
- **Output**: Color corrected images
- **Features**: KJNodes-compatible algorithms, multi-segment workflow support, adaptive processing detection

#### **Video Segment Saver** 
Professional video segment export with execution tracking.
- **Input**: Images, segment info, format settings
- **Output**: Video file path, metadata
- **Features**: Smart caching, multiple formats (MP4, WebM, MOV, AVI)

#### **Video Segment Combiner**
Automatic segment discovery and combination by execution ID.
- **Input**: Execution ID, output settings
- **Output**: Combined video, preview frames  
- **Features**: Auto-discovery, consistency validation, FFmpeg optimization

#### **Video Saver**
High-quality video export with professional encoding options.
- **Input**: Images, FPS, format, codec
- **Output**: Video file path
- **Features**: H264/H265/VP9/ProRes support, custom directories

#### **Video Upscaler**
Memory-efficient stream upscaling without VRAM limits.
- **Input**: Video file path, upscale model, factor
- **Output**: Upscaled video path, preview
- **Features**: Frame streaming, progress tracking, multiple tile sizes

#### **Video Frame Interpolation**
RIFE-powered frame interpolation for smooth motion.
- **Input**: Video file path, RIFE model, interpolation factor
- **Output**: Interpolated video, preview
- **Features**: Multiple RIFE versions, memory management, streaming processing

#### **Video Preview**
Generate preview frames from video tensors.
- **Input**: Images
- **Output**: Preview image grid
- **Features**: Automatic grid layout, quality optimization

### Utility Nodes

#### **Face Frame Detector**
Advanced face detection with eye state analysis.
- **Input**: Images, detection settings, thresholds  
- **Output**: Best face frame, frames from end, metadata, debug overlay
- **Features**: MediaPipe integration, eye aspect ratio analysis, frontal face detection

#### **Batch Trimmer**
Remove frames from end of image batches.
- **Input**: Images, frames to remove
- **Output**: Trimmed images
- **Features**: Safe validation, debug output control

#### **Resolution Picker**
Intelligent resolution selection for T2V/I2V workflows.
- **Input**: Mode toggle, batch size, image (I2V), presets
- **Output**: Mode, width, height, processed image
- **Features**: WAN-optimized presets, aspect ratio preservation, pixel count targeting

#### **Generation Settings**
Runtime segment and total length configuration.
- **Input**: Segment length, total length
- **Output**: Validated segment length, total length, loop count
- **Features**: Divisibility validation, JavaScript UI enhancements

#### **String Splitter**
Split text strings with custom delimiters.
- **Input**: Text, delimiter
- **Output**: Packed string array
- **Features**: Conflict-safe packing, whitespace handling

#### **Get String By Index**
Extract string from packed string array by index.
- **Input**: Packed strings, index
- **Output**: Selected string
- **Features**: Safe bounds checking, empty fallback

## Advanced Features

### Debug System
All nodes support environment-controlled debug output:
```bash
# Enable debug output
export DAXNODES_DEBUG=true

# Disable debug output  
export DAXNODES_DEBUG=false
```

### Metadata Preservation
Video nodes support optional metadata preservation:
- **Enabled**: Preserves all original video metadata
- **Disabled**: Strips metadata for smaller file sizes (uses `-map_metadata -1`)

### Memory Management
- **Stream Processing**: Handle unlimited video length without VRAM constraints
- **Dynamic Tiling**: Automatic tile size adjustment based on available memory
- **Batch Optimization**: Smart batch sizing for optimal performance

### Performance Optimization
- **Auto-Detection**: System capability detection for optimal settings
- **Progressive Processing**: Frame-by-frame processing for large videos  
- **FFmpeg Integration**: Hardware-accelerated encoding when available

## Categories

All nodes are organized under the **DaxNodes** category:
- `DaxNodes/Video` - Video processing nodes
- `DaxNodes/Utilities` - Utility and helper nodes

## Workflow Integration

### Basic Video Processing
```
Load Video → Video Color Correction → Video Upscaler → Video Saver
```

### Segmented Generation  
```
Generation Settings → Video Segment Saver → Video Segment Combiner
```

### Face-Aware Processing
```
Load Images → Face Frame Detector → Video Color Correction
```

## Error Handling

Professional error handling with:
- Input validation and safe defaults
- Graceful degradation when dependencies missing
- Clear error messages with resolution steps
- Automatic fallbacks for optimal user experience

## Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Windows
choco install ffmpeg
# or download from https://ffmpeg.org/download.html

# macOS  
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**MediaPipe not found**
```bash
pip install mediapipe
```

**OpenCV issues**
```bash
pip install opencv-python
```

**RIFE models missing**
Models auto-download on first use. Ensure internet connectivity.

**Memory errors during upscaling**
- Reduce batch size in upscaler settings
- Use smaller tile sizes
- Enable debug output to monitor memory usage

### Debug Output

Enable debug mode to see detailed processing information:
```bash
export DAXNODES_DEBUG=true
```

Output shows:
- Memory usage statistics
- Processing progress
- Error details and stack traces  
- Performance metrics

## License

MIT License - see [LICENSE](LICENSE) file for details.

### Third-Party Acknowledgments

- **ComfyUI**: Model loading conventions (GPL v3)
- **ComfyUI-Frame-Interpolation**: Auto-download approach (MIT)  
- **RIFE**: Video frame interpolation algorithm (Research)
- **MediaPipe**: Face detection and analysis (Apache 2.0)

## Support

For issues and feature requests, please use the GitHub issue tracker.
=======
# DaxNodes
Purpose built custom nodes for various use cases.
>>>>>>> 6e7053f529ae314e0d99cb9092da874f0ee61f5b
