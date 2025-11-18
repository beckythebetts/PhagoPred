from magicgui import magicgui
import napari
import numpy as np
from scipy import ndimage

def get_current_step_array(layer):
    """Return NumPy array of current time step from a Napari layer."""
    # t = layer.current_step[0] if hasattr(layer, "current_step") else 0
    t = napari.current_viewer().dims.current_step[0]
    data = layer.data[t] if hasattr(layer.data, "__getitem__") else layer.data
    if hasattr(data, "compute"):  # Dask array
        data = data.compute()
    return np.array(data)

def distance_bands_widget():
    @magicgui(call_button="Generate Bands",
            n_bands={"widget_type": "SpinBox", "min": 2, "max": 10, "step": 1},
            cell_label={"widget_type": "SpinBox", "min": 1, "max": 1000, "step": 1})
    def distance_bands(mask: napari.layers.Labels, n_bands: int = 3, cell_label: int = 1):
        """
        Generates normalized radial bands for a selected cell in a mask.
        """
        if mask is None:
            print("Please provide a mask image!")
            return

        viewer = napari.current_viewer()
        mask_np = get_current_step_array(mask)

        cell_mask = (mask_np == cell_label+1).astype(np.uint8)
        if cell_mask.sum() == 0:
            print(f"Cell label {cell_label} not found.")
            return

        # distance transform
        distance = ndimage.distance_transform_edt(cell_mask)
        max_dist = distance.max()
        distance_norm = distance / max_dist if max_dist > 0 else distance

        # create band image
        band_image = np.zeros_like(mask_np, dtype=np.uint8)
        for i in range(n_bands):
            lower = i / n_bands
            upper = (i + 1) / n_bands
            band_mask = (distance_norm >= lower) & (distance_norm < upper) & (cell_mask > 0)
            band_image[band_mask] = i + 1

        viewer.add_labels(band_image, name='distance bands')
        
        return band_image
    return distance_bands
