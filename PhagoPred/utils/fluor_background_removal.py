import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QSpinBox
)
from qtpy.QtCore import Qt
import scipy.ndimage

# --- Background removal functions ---
def replace_hot_pixels(image: np.ndarray,
                       upper_percentile: float = 99.9,
                       filter_size: int = 3) -> np.ndarray:
    """
    Replace the brightest pixels (above a given percentile)
    with local median values.
    """
    img = image.astype(np.float32)

    cutoff = np.percentile(img, upper_percentile)

    hot_mask = img > cutoff

    median_img = scipy.ndimage.median_filter(img, size=filter_size)

    corrected = img.copy()
    corrected[hot_mask] = median_img[hot_mask]

    return corrected

def bg_removal(img, sigma_bg=50, size_median=3):
    bg_estimate = scipy.ndimage.gaussian_filter(img, sigma=sigma_bg)
    img = img - bg_estimate
    img = np.clip(img, 0, None)
    img = scipy.ndimage.median_filter(img, size=size_median)
    return img


def bg_removal_v1(img, sigma_bg=50, sigma_smooth=3, size_median=3):
    bg_estimate = scipy.ndimage.gaussian_filter(img, sigma=sigma_bg)
    img = img - bg_estimate
    img = np.clip(img, 0, None)
    img = scipy.ndimage.median_filter(img, size=size_median)
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma_smooth)
    return img

def bg_removal_v2(img, sigma_bg=50, sigma_smooth=3, size_median=10):
    img = scipy.ndimage.median_filter(img, size=size_median)
    bg_estimate = scipy.ndimage.gaussian_filter(img, sigma=sigma_bg)
    img = img - bg_estimate
    img = np.clip(img, 0, None)
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma_smooth)
    return img

def bg_removal_v3(img, sigma_bg=20, sigma_smooth=2, size_median=5):
    bg_estimate = scipy.ndimage.gaussian_filter(img, sigma=sigma_bg)
    img = img - bg_estimate
    img = np.clip(img, 0, None)
    img = scipy.ndimage.median_filter(img, size=size_median)
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma_smooth)
    return img

def bg_removal_v4(img, sigma_bg=40, sigma_smooth=3, size_median=10):
    bg_estimate = scipy.ndimage.gaussian_filter(img, sigma=sigma_bg)
    img = img - bg_estimate
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma_smooth)
    img = np.clip(img, 0, None)
    img = scipy.ndimage.median_filter(img, size=size_median)
    return img


# --- Create custom QWidget for controls ---
class BGRemovalWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.processed_layer = None

        layout = QVBoxLayout()

        # Dropdown for method
        self.method_box = QComboBox()
        self.method_box.addItems(["v0", "v1", "v2", "v3", "v4"])
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_box)

        # Sigma BG
        layout.addWidget(QLabel("Sigma (background):"))
        self.sigma_bg = QSpinBox()
        self.sigma_bg.setRange(1, 100)
        self.sigma_bg.setValue(50)
        layout.addWidget(self.sigma_bg)

        # Sigma Smooth
        layout.addWidget(QLabel("Sigma (smoothing):"))
        self.sigma_smooth = QSpinBox()
        self.sigma_smooth.setRange(0, 20)
        self.sigma_smooth.setValue(0)
        layout.addWidget(self.sigma_smooth)

        # Median Size
        layout.addWidget(QLabel("Median size:"))
        self.size_median = QSpinBox()
        self.size_median.setRange(1, 20)
        self.size_median.setValue(3)
        layout.addWidget(self.size_median)

        # Apply button
        self.apply_button = QPushButton("Apply Background Removal")
        self.apply_button.clicked.connect(self.apply_processing)
        layout.addWidget(self.apply_button)

        layout.addStretch()
        self.setLayout(layout)

    def apply_processing(self):
        """Apply currently selected settings for background removal."""
        img_layer = self.viewer.layers.selection.active

        img = img_layer.data.astype(float)
        method = self.method_box.currentText()
        sigma_bg = self.sigma_bg.value()
        sigma_smooth = self.sigma_smooth.value()
        size_median = self.size_median.value()

        img = replace_hot_pixels(img)
        if method == "v1":
            result = bg_removal_v1(img, sigma_bg, sigma_smooth, size_median)
        elif method == "v2":
            result = bg_removal_v2(img, sigma_bg, sigma_smooth, size_median)
        elif method == "v3":
            result = bg_removal_v3(img, sigma_bg, sigma_smooth, size_median)
        elif method =='v0':
            result = bg_removal(img, sigma_bg, size_median)
        else:
            result = bg_removal_v4(img, sigma_bg, sigma_smooth, size_median)

        self.processed_layer = self.viewer.add_image(result, name=f"{method}, {sigma_bg}, {sigma_smooth}, {size_median}", blending="additive", colormap='red')

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(img_layer)
        print(f"Applied {method} with σ_bg={sigma_bg}, σ_smooth={sigma_smooth}, median={size_median}")


# --- Run Napari with the custom widget ---
if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = BGRemovalWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    print("\n💡 Drag an image into Napari, then adjust parameters and click 'Apply Background Removal'.")
    napari.run()