from qtpy import QtWidgets
import numpy as np

class LoadingBarDialog(QtWidgets.QDialog):
    def __init__(self, max_value, message="Loading..."):
        super().__init__()
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setFixedSize(300, 100)

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(message)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, max_value)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def update_progress(self, value, message=None):
        self.progress.setValue(value)
        if message:
            self.label.setText(message)
        QtWidgets.QApplication.processEvents()  # Important: refresh UI

class AllCellsView:
    def __init__(self, viewer):
        self.viewer = viewer
        self.phase_layer = None
        self.mask_layer = None
        self.points_layer = None

    def update_frame(self, frame_idx, data):
        phase_rgb = data['phase_rgb']
        mask = data['mask']
        points = data['points']
        labels = data['labels']

        if self.phase_layer is None:
            self.phase_layer = self.viewer.add_image(np.expand_dims(phase_rgb, 0), rgb=True, name="Phase Images")
        else:
            self.phase_layer.data = np.concatenate([self.phase_layer.data, np.expand_dims(phase_rgb, 0)], axis=0)

        if self.mask_layer is None:
            self.mask_layer = self.viewer.add_labels(np.expand_dims(mask, 0), name="Cell Masks", opacity=0.5)
            self.mask_layer.contour = 5
        else:
            self.mask_layer.data = np.concatenate([self.mask_layer.data, np.expand_dims(mask, 0)], axis=0)

        if self.points_layer is None:
            self.points_layer = self.viewer.add_points(points, properties={"track_id": labels}, size=5,
                                                       face_color='transparent', edge_color='yellow', text={'string':'{track_id}'})
        else:
            # Append points and properties:
            new_points = np.concatenate([self.points_layer.data, points], axis=0)
            new_labels = np.concatenate([self.points_layer.properties['track_id'], labels], axis=0)
            self.points_layer.data = new_points
            self.points_layer.properties = {"track_id": new_labels}
