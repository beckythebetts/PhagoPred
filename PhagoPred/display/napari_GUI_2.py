import numpy as np
import h5py
import torch
import napari
from qtpy.QtCore import QThread, Signal

from PhagoPred import SETTINGS

# Assume SETTINGS.DATASET points to your HDF5 file path

class FrameLoaderThread(QThread):
    frameLoaded = Signal(int, np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # frame idx, phase_rgb, mask, points, labels

    def __init__(self, start_frame, end_frame, interval=1, parent=None):
        super().__init__(parent)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.interval = interval

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with h5py.File(SETTINGS.DATASET, 'r') as f:
            phase_data = f['Images']['Phase']
            mask_data = f['Segmentations']['Phase']
            X_centers = f['Cells']['Phase']['X']
            Y_centers = f['Cells']['Phase']['Y']

            num_cells = X_centers.shape[1]

            # Color LUT for cell labels (including background color at idx 0)
            # LUT = torch.randint(low=10, high=255, size=(num_cells, 3), dtype=torch.uint8).to(device)
            # lut_for_neg1 = torch.tensor([[0, 0, 0]], device=device, dtype=torch.uint8)
            # LUT = torch.cat([lut_for_neg1, LUT], dim=0)
            # lut_np = LUT.cpu().numpy() / 255.0

            frames = np.arange(self.start_frame, self.end_frame, self.interval)

            for i, frame in enumerate(frames):
                phase_im = phase_data[frame]
                mask = mask_data[frame]

                # Convert grayscale phase to RGB uint8
                phase_rgb = np.stack([phase_im]*3, axis=-1).astype(np.uint8)

                # Prepare points and labels for cell centers
                track_ids = np.where(~np.isnan(X_centers[frame]))[0]

                points = []
                labels = []
                for cell_idx in track_ids:
                    x = X_centers[frame, cell_idx]
                    y = Y_centers[frame, cell_idx]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    # napari expects (frame, y, x)
                    points.append([i, y, x])
                    labels.append(cell_idx)

                points = np.array(points) if points else np.empty((0, 3))
                labels = np.array(labels) if labels else np.empty((0,))

                self.frameLoaded.emit(frame, phase_rgb, mask, points, labels)


class ViewerApp:
    def __init__(self, viewer):
        self.viewer = viewer
        self.thread = FrameLoaderThread(start_frame=0, end_frame=100, interval=1)
        self.thread.frameLoaded.connect(self.add_frame_to_viewer)
        self.thread.start()

        # Prepare empty layers placeholders
        self.phase_layer = None
        self.labels_layer = None
        self.points_layer = None

    def add_frame_to_viewer(self, frame, phase_rgb, mask, points, labels):
        if self.phase_layer is None:
            # Initialize 4D image layer: (time, height, width, channels)
            self.phase_layer = self.viewer.add_image(
                np.expand_dims(phase_rgb, axis=0), 
                rgb=True, 
                name="Phase Images"
            )
        else:
            # Append frame to existing layer
            self.phase_layer.data = np.concatenate(
                [self.phase_layer.data, np.expand_dims(phase_rgb, axis=0)],
                axis=0)

        if self.labels_layer is None:
            self.labels_layer = self.viewer.add_labels(
                np.expand_dims(mask, axis=0),
                name="Cell Masks",
                opacity=0.5,
                blending='additive',
                # contour=10,
            )
            self.labels_layer.contour = 5
        else:
            self.labels_layer.data = np.concatenate(
                [self.labels_layer.data, np.expand_dims(mask, axis=0)],
                axis=0)

        if self.points_layer is None:
            properties = {
                "track_id": labels.astype(int)
            }
            text_props = {
                'string': '{track_id}',
                'size': 12,
                'anchor': 'center',
                'translation': [0, 0, 0],
                'color': 'yellow',
            }
            self.points_layer = self.viewer.add_points(
                points,
                properties=properties,
                size=5,
                face_color='transparent',
                border_color='transparent',
                text=text_props,
                name='Cell IDs',
            )
        else:
            # Append new points and labels, updating properties
            old_points = self.points_layer.data
            old_props = self.points_layer.properties

            new_points = np.concatenate([old_points, points], axis=0) if points.size else old_points
            new_labels = np.concatenate([old_props['track_id'], labels]) if labels.size else old_props['track_id']

            self.points_layer.data = new_points
            self.points_layer.properties = {"track_id": new_labels}

# Example usage:
viewer = napari.Viewer(show=False)
app = ViewerApp(viewer)
napari.run()
