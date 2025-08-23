# RIFE Models Directory

## Auto-Download

The RIFE node automatically downloads models on first use from the same trusted sources as ComfyUI-Frame-Interpolation:

- **rife47.pth** - Stable, widely used
- **rife48.pth** - Improved version  
- **rife49.pth** - Latest stable

Models are downloaded from established GitHub releases used by the ComfyUI community.

## File Structure

After auto-download, your directory will look like:
```
DaxNodes/models/rife/
├── README.md (this file)
├── rife47.pth
├── rife48.pth
└── rife49.pth
```

## Download Sources

Uses the same trusted repositories as ComfyUI-Frame-Interpolation:
- styler00dollar/VSGAN-tensorrt-docker/releases
- Fannovel16/ComfyUI-Frame-Interpolation/releases  
- dajes/frame-interpolation-pytorch/releases

## ComfyUI-Frame-Interpolation Integration

If you already have ComfyUI-Frame-Interpolation installed, our RIFE node will automatically detect and use their RIFE models! Models from ComfyUI-Frame-Interpolation will appear in the dropdown with `[ComfyUI-FI]` prefix.

## Notes

- **Auto-detects ComfyUI-Frame-Interpolation models** - no need to duplicate downloads
- Uses same download mechanism as ComfyUI-Frame-Interpolation extension
- Auto-download happens on first node execution
- Enhanced linear interpolation fallback if download fails
- Models are loaded on-demand and cached for performance