from magicgui import magicgui
import napari

@magicgui(call_button="Split Channels")
def split_channels_widget(layer: napari.layers.Image):
    data = layer.data
    viewer = napari.current_viewer()

    if data.ndim < 3:
        raise ValueError("Layer does not have a channel dimension")

    for c in range(data.shape[0]): 
        viewer.add_image(data[c], name=f"{layer.name}_ch{c}")

    return None