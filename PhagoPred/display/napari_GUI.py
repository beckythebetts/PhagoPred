from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QToolBar, QLineEdit, QAction, QDialog, QLabel, QProgressBar, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
from qtpy.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from napari import Viewer
import napari
import numpy as np
import h5py
import torch
import sys
import pyqtgraph as pg

import random

from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

class LoadingBarDialog(QDialog):
    def __init__(self, max_value, message="Loading..."):
        super().__init__()
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()

        self.label = QLabel(message)
        self.progress = QProgressBar()
        self.progress.setRange(0, max_value)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def update_progress(self, value, message=None):
        self.progress.setValue(value)
        if message:
            self.label.setText(message)
        QApplication.processEvents()  # Important: refresh UI

class ToolBar(QToolBar):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent

        # Button for next cell
        self.next_cell_action = QAction('Next Cell', self)
        self.next_cell_action.setStatusTip("Go to next cell")
        self.next_cell_action.triggered.connect(self.parent.next_cell)  # Call main window method

        self.addAction(self.next_cell_action)

        self.addSeparator()

        label = QLabel("Enter Cell Index:")
        label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.addWidget(label)

        # Text box for entering cell index
        self.cell_index_edit = QLineEdit()
        self.cell_index_edit.setFixedWidth(40)  # Optional: fix width for neatness
        self.cell_index_edit.setPlaceholderText("Enter index")
        self.cell_index_edit.setAlignment(Qt.AlignCenter)
        self.cell_index_edit.returnPressed.connect(self.load_cell_from_text)
        self.addWidget(self.cell_index_edit)

    def load_cell_from_text(self):
        text = self.cell_index_edit.text()
        try:
            cell_idx = int(text)
            self.parent.cell_idx = cell_idx
            self.parent.load_cell()
        except ValueError:
            # Handle invalid input gracefully (e.g., ignore or show a message)
            pass

# class AllCellsViewer(QMainWindow):

#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("All Cells Viewer")
#         self.setMinimumSize(300, 300)
#         self.setMaximumSize(300, 300)
#         self.resize(1000, 300)

#         self.phase_images = None

#         self.start_frame = 0
#         self.end_frame = 200

#         self.interval = 1 # Only show every {self.interval} frames (for speed and memory)

#         self.viewer = napari.Viewer(show=False)

#             # Central widget
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)

#         # Layout
#         layout = QHBoxLayout(central_widget)
#         # layout = QHBoxLayout()
#         layout.addWidget(self.viewer.window._qt_viewer)

#         self.load_images()
        


#     def load_images(self):
#         hdf5_file = SETTINGS.DATASET
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         with h5py.File(hdf5_file, 'r') as f:
#             num_cells = f['Cells']['Phase']['X'].shape[1]

#             LUT = torch.randint(low=10, high=255, size=(num_cells, 3), dtype=torch.uint8).to(device)
#             lut_for_neg1 = torch.tensor([[0, 0, 0]], device=device, dtype=torch.uint8)
#             LUT = torch.cat([lut_for_neg1, LUT], dim=0)

#             phase_data = f['Images']['Phase']
#             mask_data = f['Segmentations']['Phase']

#             if self.start_frame is None:
#                 self.start_frame = 0
#             if self.end_frame is None:
#                 self.end_frame = phase_data.shape[0]

#             display_frames = np.arange(start=self.start_frame, stop=self.end_frame, step=self.interval)

#             loading_bar = LoadingBarDialog(np.max(display_frames), message="Loading images...")
#             loading_bar.show()

#             images = []

#             for frame in display_frames:
#                 phase_im = np.stack([phase_data[frame]]*3, axis=-1)
#                 mask = mask_data[frame]

#                 phase_im = torch.tensor(phase_im, dtype=torch.uint8).to(device)
#                 mask = torch.tensor(mask).to(device)

#                 outlines = mask_funcs.mask_outlines(mask, thickness=3).type(torch.int64)
#                 # print(outlines.shape)
#                 colour_outlines = (LUT[outlines]).type(torch.uint8)
#                 # print(colour_outlines.shape)
#                 # print(phase_im.shape)

#                 outlined_phase_image =  (torch.where(outlines.unsqueeze(-1).expand_as(colour_outlines)>0, colour_outlines, phase_im))
#                 outlined_phase_image = outlined_phase_image.cpu().numpy() / 256

#                 images.append(outlined_phase_image)
#                 loading_bar.update_progress(frame)
#             images = np.stack(images, axis=0)
#             self.viewer.add_image(images, rgb=True)

class AllCellsViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("All Cells Viewer")
        self.setMinimumSize(300, 300)
        self.setMaximumSize(300, 300)
        self.resize(1000, 300)

        self.phase_images = None
        self.start_frame = 0
        self.end_frame = 200
        self.interval = 1  # Only show every {self.interval} frames (for speed and memory)

        self.viewer = napari.Viewer(show=False)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.addWidget(self.viewer.window._qt_viewer)

        self.load_images()

    def load_images(self):
        hdf5_file = SETTINGS.DATASET
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with h5py.File(hdf5_file, 'r') as f:
            num_cells = f['Cells']['Phase']['X'].shape[1]

            # Keep random colors per session
            LUT = torch.randint(low=10, high=255, size=(num_cells, 3), dtype=torch.uint8).to(device)
            lut_for_neg1 = torch.tensor([[0, 0, 0]], device=device, dtype=torch.uint8)
            LUT = torch.cat([lut_for_neg1, LUT], dim=0)

            phase_data = f['Images']['Phase']
            mask_data = f['Segmentations']['Phase']
            X_centers = f['Cells']['Phase']['X']
            Y_centers = f['Cells']['Phase']['Y']

            if self.start_frame is None:
                self.start_frame = 0
            if self.end_frame is None:
                self.end_frame = phase_data.shape[0]

            display_frames = np.arange(start=self.start_frame, stop=self.end_frame, step=self.interval)

            loading_bar = LoadingBarDialog(len(display_frames), message="Loading images...")
            loading_bar.show()

            images = []
            points = []
            labels = []

            text = {
                'string': '{track_id}',         # match the property name exactly
                'size': 12,
                'anchor': 'center',
                'translation': [0, 0, 0],          # you can keep this or tweak to shift text
                'color': "yellow",
            }

            for i, frame in enumerate(display_frames):
                phase_im = np.stack([phase_data[frame]] * 3, axis=-1)
                mask = mask_data[frame]

                phase_im = torch.tensor(phase_im, dtype=torch.uint8).to(device)
                mask = torch.tensor(mask).to(device)

                outlines = mask_funcs.mask_outlines(mask, thickness=3).type(torch.int64)
                colour_outlines = (LUT[outlines]).type(torch.uint8)

                outlined_phase_image = torch.where(
                    outlines.unsqueeze(-1).expand_as(colour_outlines) > 0,
                    colour_outlines,
                    phase_im
                )
                outlined_phase_image = outlined_phase_image.cpu().numpy() / 256
                images.append(outlined_phase_image)

                # === Use precomputed centers + track ids ===
                track_ids = np.where(~np.isnan(X_centers[frame]))[0]

                for cell_idx in track_ids:
                    x = X_centers[frame, cell_idx]
                    y = Y_centers[frame, cell_idx]

                    if np.isnan(x) or np.isnan(y):
                        continue

                    points.append([i, x, y])  # Napari: (T, Y, X)
                    labels.append(str(cell_idx))
                loading_bar.update_progress(i)

            images = np.stack(images, axis=0)
            self.viewer.add_image(images, rgb=True, name="Cells")

            properties = {
                "track_id": np.array([int(label) for label in labels]),
            }

            if points:
                self.viewer.add_points(
                    points,
                    properties=properties,
                    size=1,
                    face_color='transparent',
                    edge_color='transparent',
                    text=text,
                    name='Cell IDs',
                )


class CellViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setMinimumSize(300, 300)
        self.resize(1000, 300)

        self.cell_idx = 0
        self.first_frame = None
        self.last_frame = None
        self.frame_size = 300

        self.phase_data = None
        self.epi_data = None
        self.cell_outline = None

        self.feature_names=[
                # 'Total Fluorescence', 
                # 'Fluorescence Distance Mean', 
                # 'Fluorescence Distance Variance'
                'Area',
                # 'Circularity',
                # 'Perimeter',
                # 'Displacement',
                # 'Speed',
                # 'Mode 0',
                # 'Mode 1',
                # 'Mode 2',
                # 'Mode 3',
                # 'Speed',
                # 'Phagocytes within 100 pixels',
                # 'Phagocytes within 250 pixels',
                # 'Phagocytes within 500 pixels',
                'X',
                'Y',
                # 'CellDeath'
        ]
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QHBoxLayout(central_widget)

        # Napari viewer
        self.viewer = napari.Viewer(show=False)
        self.get_feature_plot_widget()
        self.toolBar = ToolBar(self)

        self.addToolBar(self.toolBar)

        self.load_cell()

        layout.addWidget(self.viewer.window._qt_viewer)
        layout.addWidget(self.plot_widget)
        self.setup_frame_display()

        self.viewer.dims.events.current_step.connect(self.update_vertical_lines)

    def load_cell(self):
        hdf5_file = SETTINGS.DATASET
        
        with h5py.File(hdf5_file, 'r') as f:
            self.loading_dialog = LoadingBarDialog(0, 'Reading dataset...')
            self.first_frame, self.last_frame = tools.get_cell_end_frames(self.cell_idx, f)
            self.loading_dialog.close()

            self.get_cell_images(f)
            self.update_feature_plots(f)
            self.cell_death = f['Cells']['Phase']['CellDeath'][0, self.cell_idx]
            if np.isnan(self.cell_death):
                self.cell_death_str = 'Alive'
            else:
                self.cell_death_str = f'Cell Death at frame {int(self.cell_death)}'
        
        self.add_vertical_lines_to_plots()

        self.viewer.layers.clear()
        self.viewer.add_image(self.phase_data, name='Phase')
        self.viewer.add_image(self.epi_data, name='Epi', blending='additive', colormap='red', opacity=0.5)

        self.viewer.add_image((self.cell_outline*255).astype(np.uint8), name='Outline', colormap='yellow', opacity=0.8, blending='additive')
        self.viewer.dims.set_current_step(0, 0)


        self.setWindowTitle(f"Cell Viewer - Cell {self.cell_idx}, {self.cell_death_str}")

        # self.loading_dialog.close()

    def get_cell_images(self, f: h5py.File):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        n_frames = self.last_frame - self.first_frame
        phase_data = np.empty((n_frames, self.frame_size, self.frame_size))
        epi_data = np.empty((n_frames, self.frame_size, self.frame_size))
        mask = np.empty((n_frames, self.frame_size, self.frame_size), dtype=np.int32)

        # Start progress dialog
        self.loading_dialog = LoadingBarDialog(n_frames, message="Loading cell images...")
        self.loading_dialog.show()

        x_centres = f['Cells']['Phase']['X'][self.first_frame:self.last_frame, self.cell_idx]
        y_centres = f['Cells']['Phase']['Y'][self.first_frame:self.last_frame, self.cell_idx]
        x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

        phase_stack = f['Images']['Phase'][self.first_frame:self.last_frame]
        epi_stack = f['Images']['Epi'][self.first_frame:self.last_frame]
        mask_stack = f['Segmentations']['Phase'][self.first_frame:self.last_frame]

        for idx in range(n_frames):
            xmin, xmax, ymin, ymax = mask_funcs.get_crop_indices(
                (y_centres[idx], x_centres[idx]), self.frame_size, SETTINGS.IMAGE_SIZE
            )
            phase_data[idx] = phase_stack[idx][ymin:ymax, xmin:xmax]
            epi_data[idx] = epi_stack[idx][ymin:ymax, xmin:xmax]
            mask[idx] = mask_stack[idx][ymin:ymax, xmin:xmax]

            self.loading_dialog.update_progress(idx+1)

        cell_mask = (mask == self.cell_idx)

        if not cell_mask.any():
            raise Exception(f'Cell of index {self.cell_idx} not found')
        
        epi_data = tools.threshold_image(epi_data)
        # contours = skimage.measure.find_contours(cell_mask, level=0.5)
        cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()

        # Threshold epi_data (assuming your existing tool does that)
        epi_data = tools.threshold_image(epi_data)

        self.phase_data = phase_data
        self.epi_data = epi_data
        self.cell_outline = cell_outline

        self.loading_dialog.close()

    
    def get_feature_plot_widget(self) -> None:

        # Create QWidget to hold all plots
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.plot_widgets = []  # Store for later updates

        for feature_name in self.feature_names:
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setLabel('left', feature_name)
            # plot_widget.setLabel('bottom', 'Frames')

            layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)

        self.plot_widget = widget 

    def update_feature_plots(self, f: h5py.File):
        self.loading_dialog = LoadingBarDialog(max_value=len(self.feature_names), message='Updating plots...')
        for i, feature_name in enumerate(self.feature_names):
            y_data = f['Cells']['Phase'][feature_name][self.first_frame:self.last_frame, self.cell_idx]
            x_data = np.arange(self.first_frame, self.last_frame)

            plot_item = self.plot_widgets[i].getPlotItem()
            plot_item.clear()
            plot_item.plot(x_data, y_data, pen=pg.mkPen(color='k'))
            self.loading_dialog.update_progress(i+1)
        self.loading_dialog.close()
    
    def next_cell(self):
        self.cell_idx += 1
        self.load_cell()

    def setup_frame_display(self):

        def update_frame_label(event):
            # event gives you current indices along all dims, for time usually dim=0
            current_frame = self.viewer.dims.current_step[0]
            actual_frame = self.first_frame + current_frame  # Adjust for your offset
            self.statusBar().showMessage(f"Current frame: {actual_frame}")

        self.viewer.dims.events.current_step.connect(update_frame_label)
        
        # Initialize the label immediately
        update_frame_label(None)

    def add_vertical_lines_to_plots(self):
        self.vlines = []
        for plot_widget in self.plot_widgets:
            vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
            plot_widget.addItem(vline)
            self.vlines.append(vline)

    def update_vertical_lines(self, event):
        # Current frame relative to loaded frames
        current_rel_frame = self.viewer.dims.current_step[0]
        # Map to actual frame number (optional, if needed)
        actual_frame = self.first_frame + current_rel_frame

        # Move each vertical line to current_rel_frame position
        for vline in self.vlines:
            vline.setPos(actual_frame)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        super().keyPressEvent(event)
    
def main():
    app = QApplication(sys.argv)
    cell_window = CellViewer()
    all_window = AllCellsViewer()
    # window.show()
    cell_window.showMaximized()
    all_window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()