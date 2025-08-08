from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import (QToolBar, QLineEdit, QAction, QDialog, QLabel, QProgressBar, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QInputDialog,
                            QFormLayout,
                            QSpinBox,
                            QDialogButtonBox,
                            QMessageBox,
                            QStatusBar,
                            )

from qtpy.QtCore import Qt, QTimer, QThread, Signal
import matplotlib.pyplot as plt
from napari import Viewer
import napari
import numpy as np
import h5py
import torch
import sys
import pyqtgraph as pg

from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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

    def show(self):
        super().show()
        QApplication.processEvents()

    def update_progress(self, value, message=None):
        self.progress.setValue(value)
        if message:
            self.label.setText(message)
        QApplication.processEvents()  # Important: refresh UI

class FrameSliceDialog(QDialog):
    def __init__(self, start=0, end=100, interval=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Frame Slice")

        layout = QVBoxLayout(self)

        # Form with spin boxes
        form_layout = QFormLayout()

        self.start_input = QSpinBox()
        self.start_input.setMinimum(0)
        self.start_input.setMaximum(1e5)
        self.start_input.setValue(start)

        self.end_input = QSpinBox()
        self.end_input.setMinimum(1)
        self.end_input.setMaximum(1e5)
        self.end_input.setValue(end)

        self.interval_input = QSpinBox()
        self.interval_input.setMinimum(1)
        self.interval_input.setMaximum(1e4)
        self.interval_input.setValue(interval)

        form_layout.addRow("Start Frame:", self.start_input)
        form_layout.addRow("End Frame:", self.end_input)
        form_layout.addRow("Interval:", self.interval_input)

        layout.addLayout(form_layout)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def get_values(self):
        return (
            self.start_input.value(),
            self.end_input.value(),
            self.interval_input.value()
        )
    
class CellToolBar(QToolBar):

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

class AllCellToolBar(QToolBar):
    def __init__(self, parent=None):

        super().__init__()
        self.parent = parent

        self.toggle_outline_action = QAction("Toggle Outlines")
        self.toggle_outline_action.triggered.connect(self.parent.show_outlines)
        self.addAction(self.toggle_outline_action)

        self.load_slice_action = (QAction("Load Frames"))
        self.load_slice_action.triggered.connect(self.parent.load_frame_slice_dialog)
        self.addAction(self.load_slice_action)


class AllCellsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStatusBar(QStatusBar())
        self.setWindowTitle("All Cells Viewer")
        # self.setMinimumSize(300, 300)

        # self.setMaximumSize(300, 300)
        # self.resize(1000, 300)
        self.tool_bar = AllCellToolBar(self)
        self.addToolBar(self.tool_bar)

        self.phase_images = None
        self.start_frame = 0
        self.end_frame = 20
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

        with h5py.File(hdf5_file, 'r') as f:
   
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
            labels = []
            masks = [] 

            points = []

            text = {
                'string': '{track_id}',         # match the property name exactly
                'size': 12,
                'anchor': 'center',
                'translation': [0, 0, 0],          # you can keep this or tweak to shift text
                'color': "m",
            }

            for i, frame in enumerate(display_frames):
                phase_im = phase_data[frame]
                images.append(phase_im)

                mask = mask_data[frame]
                masks.append(mask)

                # === Gather points for this frame ===
                track_ids = np.where(~np.isnan(X_centers[frame]))[0]

                for cell_idx in track_ids:
                    x = X_centers[frame, cell_idx]
                    y = Y_centers[frame, cell_idx]

                    if np.isnan(x) or np.isnan(y):
                        continue

                    points.append([i, x, y])  # Note: napari points are (frame, y, x)
                    labels.append(str(cell_idx))

                loading_bar.update_progress(i)

            images = np.stack(images, axis=0)
            masks = np.stack(masks, axis=0)

            self.viewer.add_image(images, name="Phase Images")

            self.labels_layer = self.viewer.add_labels(
                masks.astype(np.uint16),
                name="Cell Masks",
                opacity=1.0,
                blending='additive',
            )
            self.labels_layer.contour = 5
            self.labels_layer.outline_colour = 'm'

            if points:
                properties = {
                    "track_id": np.array([int(label) for label in labels]),
                }

                self.viewer.add_points(
                    points,
                    properties=properties,
                    size=5,
                    face_color='transparent',
                    border_color='transparent',
                    text=text,
                    name='Cell IDs',
                )
        self.setup_frame_display()

    def show_outlines(self):
        if hasattr(self, 'labels_layer') and self.labels_layer is not None:
            self.labels_layer.visible = not self.labels_layer.visible

    def load_frame_slice_dialog(self):
        dialog = FrameSliceDialog(
        start=self.start_frame,
        end=self.end_frame,
        interval=self.interval,
        parent=self
        )
        if dialog.exec_() == QDialog.Accepted:
            start, end, interval = dialog.get_values()
            if start >= end:
                QMessageBox.warning(self, "Invalid Input", "Start frame must be less than end frame.")
                return

            self.start_frame = start
            self.end_frame = end
            self.interval = interval

            self.viewer.layers.clear()
            self.load_images()
    def setup_frame_display(self):

        def update_frame_label(event=None):
            # event gives you current indices along all dims, for time usually dim=0
            current_frame = self.viewer.dims.current_step[0]
            actual_frame = self.start_frame + (current_frame*self.interval)  # Adjust for your offset
            self.statusBar().showMessage(f"Current frame: {actual_frame}")

        self.viewer.dims.events.current_step.connect(update_frame_label)
        update_frame_label()

    def closeEvent(self, event):
        print("AllCellsViewer is closing")
        super().closeEvent(event)


class CellLoaderThread(QThread):
    progress = Signal(int)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray)  # phase_data, epi_data, cell_outline

    def __init__(self, cell_idx, first_frame, last_frame, frame_size, hdf5_file):
        super().__init__()
        self.cell_idx = cell_idx
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.frame_size = frame_size
        self.hdf5_file = hdf5_file

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_frames = self.last_frame - self.first_frame

            phase_data = np.empty((n_frames, self.frame_size, self.frame_size))
            epi_data = np.empty((n_frames, self.frame_size, self.frame_size))
            mask = np.empty((n_frames, self.frame_size, self.frame_size), dtype=np.int32)

            with h5py.File(self.hdf5_file, 'r') as f:
                x_centres = f['Cells']['Phase']['X'][self.first_frame:self.last_frame, self.cell_idx]
                y_centres = f['Cells']['Phase']['Y'][self.first_frame:self.last_frame, self.cell_idx]
                x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

                for idx in range(n_frames):
                    frame_idx = self.first_frame + idx
                    xmin, xmax, ymin, ymax = mask_funcs.get_crop_indices(
                        (y_centres[idx], x_centres[idx]), self.frame_size, SETTINGS.IMAGE_SIZE
                    )
                    phase_data[idx] =  f['Images']['Phase'][frame_idx, ymin:ymax, xmin:xmax]
                    epi_data[idx] = f['Images']['Epi'][frame_idx, ymin:ymax, xmin:xmax]
                    mask[idx] = f['Segmentations']['Phase'][frame_idx, ymin:ymax, xmin:xmax]
                    self.progress.emit(idx + 1)

            cell_mask = (mask == self.cell_idx)

            if not cell_mask.any():
                cell_outline = np.zeros_like(mask[0])
            else:
                cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()

            epi_data = tools.threshold_image(epi_data)

            self.finished.emit(phase_data, epi_data, cell_outline)

        except Exception as e:
            import traceback
            traceback.print_exc()



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
        self.toolBar = CellToolBar(self)

        self.addToolBar(self.toolBar)

        self.load_cell()

        layout.addWidget(self.viewer.window._qt_viewer)
        layout.addWidget(self.plot_widget)
        self.setup_frame_display()

        self.viewer.dims.events.current_step.connect(self.update_vertical_lines)

        

    def load_cell(self):
        hdf5_file = SETTINGS.DATASET

        with h5py.File(hdf5_file, 'r') as f:
            self.first_frame, self.last_frame = tools.get_cell_end_frames(self.cell_idx, f)

        # Show loading dialog with number of frames to load
        n_frames = self.last_frame - self.first_frame
        self.loading_dialog = LoadingBarDialog(n_frames, 'Loading cell images...')
        self.loading_dialog.show()

        # Start loader thread
        self.loader_thread = CellLoaderThread(
            cell_idx=self.cell_idx,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            frame_size=self.frame_size,
            hdf5_file=hdf5_file
        )
        self.loader_thread.progress.connect(self.loading_dialog.update_progress)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.start()


    def on_load_finished(self, phase_data, epi_data, cell_outline):
        self.phase_data = phase_data
        self.epi_data = epi_data
        self.cell_outline = cell_outline

        self.viewer.add_image(self.phase_data, name='Phase')
        self.viewer.add_image(self.epi_data, name='Epi', blending='additive', colormap='red', opacity=0.5)
        self.viewer.add_image((self.cell_outline * 255).astype(np.uint8), name='Outline', colormap='yellow', opacity=0.8, blending='additive')
        self.viewer.dims.set_current_step(0, 0)

        self.loading_dialog.close()


        hdf5_file = SETTINGS.DATASET
        with h5py.File(hdf5_file, 'r') as f:
            self.update_feature_plots(f)
            cell_death = f['Cells']['Phase']['CellDeath'][0, self.cell_idx]
            if np.isnan(cell_death):
                cell_death_str = 'Alive'
            else:
                cell_death_str = f'Cell Death at frame {int(cell_death)}'
        self.setWindowTitle(f"Cell Viewer - Cell {self.cell_idx}, {cell_death_str}")
        self.add_vertical_lines_to_plots()

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
        self.add_vertical_lines_to_plots()

    def update_feature_plots(self, f: h5py.File):
        self.loading_dialog = LoadingBarDialog(max_value=len(self.feature_names), message='Updating plots...')
        self.loading_dialog.show()
        for i, feature_name in enumerate(self.feature_names):
            y_data = f['Cells']['Phase'][feature_name][self.first_frame:self.last_frame, self.cell_idx]
            x_data = np.arange(self.first_frame, self.last_frame)

            plot_item = self.plot_widgets[i].getPlotItem()
            plot_item.clear()
            plot_item.plot(x_data, y_data, pen=pg.mkPen(color='k'))
            self.loading_dialog.update_progress(i+1)
        self.loading_dialog.close()
        self.align_feature_plots()
    
    def align_feature_plots(self):
        x_min, x_max = self.first_frame, self.last_frame

        for plot_widget in self.plot_widgets:
            plot_widget.setXRange(x_min, x_max, padding=0)
            plot_widget.enableAutoRange(enable=False, axis='x')
        
        # Link all x axes to the first plot
        for plot_widget in self.plot_widgets[1:]:
            plot_widget.setXLink(self.plot_widgets[0])

        # Hide x-axis on all but the last plot for cleaner look
        for plot_widget in self.plot_widgets[:-1]:
            plot_widget.showAxis('bottom', False)
        
        # Adjust margins (optional)
        for plot_widget in self.plot_widgets:
            plot_widget.getPlotItem().layout.setContentsMargins(5, 5, 5, 0)
            plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)

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

    def closeEvent(self, event):
        print("CellViewer is closing")
        super().closeEvent(event)
    
def main():
    app = QApplication(sys.argv)

    all_window = AllCellsViewer()
    all_window.show()
    cell_window = CellViewer()
    cell_window.show()
  


    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

