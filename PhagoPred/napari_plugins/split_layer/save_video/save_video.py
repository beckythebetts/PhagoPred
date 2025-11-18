import napari
import numpy as np
import imageio
from skimage import color

# ------------------------------
# 1. Get current viewer
# ------------------------------
viewer = napari.current_viewer()

# Find layers
layer1 = viewer.layers[0]
layer2 = viewer.layers[1]

n_frames = layer1.data.shape[0]
print(f"Detected {n_frames} time frames.")

# Identify which layer is "Epi"
epi_layer = None
other_layer = None
for layer in viewer.layers:
    if layer.name.lower() == "epi":
        epi_layer = layer
    else:
        other_layer = layer

if epi_layer is None or other_layer is None:
    raise ValueError("Could not find layer 'Epi' and another layer to merge.")

# ------------------------------
# 2. Merge frames with colormap
# ------------------------------
frames = []
alpha = 0.5  # blending factor

for t in range(50):
    if t == 182:  # skip this frame
        continue

    # Get current frame
    epi_frame = epi_layer.data[t]
    other_frame = other_layer.data[t]

    # Normalize to 0-1 for coloring
    epi_norm = (epi_frame - epi_frame.min()) / max(epi_frame.max() - epi_frame.min(), 1e-8)
    other_norm = (other_frame - other_frame.min()) / max(other_frame.max() - other_frame.min(), 1e-8)

    # Convert "Epi" layer to red colormap (R channel)
    epi_rgb = np.zeros((*epi_norm.shape, 3), dtype=np.float32)
    epi_rgb[..., 0] = epi_norm  # red channel
    # optional: could multiply by intensity to match napari opacity
    # epi_rgb[..., 0] *= epi_layer.opacity

    # Convert other layer to grayscale RGB
    other_rgb = np.stack([other_norm]*3, axis=-1)

    # Blend the two layers
    merged = (1-alpha)*other_rgb + alpha*epi_rgb
    merged = np.clip(merged*255, 0, 255).astype(np.uint8)
    merged = np.ascontiguousarray(merged)

    frames.append(merged)

print(f"Prepared {len(frames)} frames for video.")

# ------------------------------
# 3. Save as MP4
# ------------------------------
output_filename = "merged_layers_small.mp4"

imageio.mimwrite(
    output_filename,
    frames,
    fps=10,
    codec='libx264',
    quality=5,
    macro_block_size=None,
    ffmpeg_params=['-preset', 'fast', '-pix_fmt', 'yuv420p']
)

print(f"Saved video to: {output_filename}")