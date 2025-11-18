from magicgui import magicgui
import napari
import numpy as np
from scipy import ndimage

# --------------------------
# Function to compute radial bands
# --------------------------
@magicgui(call_button="Generate Bands",
          n_bands={"widget_type": "SpinBox", "min": 2, "max": 10, "step": 1},
          cell_label={"widget_type": "SpinBox", "min": 1, "max": 100, "step": 1})
def radial_bands_plugin(mask: "napari.types.ImageData", n_bands: int = 3, cell_label: int = 1):
    """
    Generates normalized radial bands for a selected cell in a mask.
    """
    if mask is None:
        print("Please provide a mask image!")
        return

    mask = mask.astype(np.int32)
    cell_mask = (mask == cell_label).astype(np.uint8)
    if cell_mask.sum() == 0:
        print(f"Cell label {cell_label} not found.")
        return

    # distance transform
    distance = ndimage.distance_transform_edt(cell_mask)
    max_dist = distance.max()
    distance_norm = distance / max_dist if max_dist > 0 else distance

    # create band image
    band_image = np.zeros_like(mask, dtype=np.uint8)
    for i in range(n_bands):
        lower = i / n_bands
        upper = (i + 1) / n_bands
        band_mask = (distance_norm >= lower) & (distance_norm < upper)
        band_image[band_mask] = i + 1

    return band_image