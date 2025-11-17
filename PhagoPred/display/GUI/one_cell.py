from pathlib import Path
import napari
import h5py
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QThread, Signal

from PhagoPred.display.GUI.plots_widget import FeaturePlotsWidget
from PhagoPred.display.GUI.utils import LoadingBarDialog
from PhagoPred.utils import tools, mask_funcs
from PhagoPred import SETTINGS

class CellViewer(QWidget):
    """Display a single cell with cropped images in Napari and feature plots."""

    cell_death_signal = Signal(int)
    def __init__(self, viewer: napari.Viewer, hdf5_file: h5py.File, cell_idx: int, frame_size=300):
        super().__init__()
        
        
        self.viewer = viewer
        self.hdf5_file = hdf5_file
        self.cell_idx = cell_idx
        self.frame_size = frame_size

        self.phase_data = None
        self.epi_data = None

        # Read frame range from file
        self.first_frame = int(self.hdf5_file['Cells']['Phase']['First Frame'][0, self.cell_idx])
        self.last_frame = int(self.hdf5_file['Cells']['Phase']['Last Frame'][0, self.cell_idx])
        self.cell_death = self.hdf5_file['Cells']['Phase']['CellDeath'][0, self.cell_idx]
        try:
            self.cell_death = int(self.cell_death)
        except ValueError:
            pass

        # Feature names from HDF5
        self.progress_bar = None
        self.feature_names = [k for k in self.hdf5_file['Cells']['Phase'].keys() if k not in ('Images','First Frame','Last Frame','X','Y', 'CellDeath', 'Macrophage', 'Dead Macrophage')]
        
        # Create plots widget with checkboxes
        self.plots_widget = FeaturePlotsWidget(self.feature_names)
        self.plots_dock = self.viewer.window.add_dock_widget(self.plots_widget, area='right')

        # Start loading images in thread
        self.loader_thread = CellLoaderThread(
            cell_idx=self.cell_idx,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            frame_size=self.frame_size,
            hdf5_file=self.hdf5_file  # pass already open hdf5
        )
        # self.loader_thread.progress.connect(self.on_progress)
        self.loader_thread.start_signal.connect(self._start_load)
        self.loader_thread.start()
        # Link Napari frame changes to vertical lines
        self.vlines = {}
        self.viewer.dims.events.current_step.connect(self.update_vertical_lines)
        
    def _start_load(self, frames):
        self.progress_bar = LoadingBarDialog(frames)
        self.progress_bar.show()
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.progress.connect(self.progress_bar.update_progress)
        
        
    def on_load_finished(self, phase_data, epi_data, mask):
        self.progress_bar.close()
        self.cell_death_signal.emit(self.cell_death)
        self.phase_data = phase_data
        self.epi_data = epi_data

        self.viewer.add_image(self.phase_data, name=f'Phase {self.cell_idx}', colormap='gray', translate=(self.first_frame, 0, 0))
        self.viewer.add_image(self.epi_data, name=f'Epi {self.cell_idx}', blending='additive', colormap='red', opacity=0.5, translate=(self.first_frame, 0, 0))
        # cell_mask = (mask == self.cell_idx).astype(np.int32)
        labels_layer = self.viewer.add_labels(mask, name=f'Cell {self.cell_idx}', colormap={1: [255, 255, 0, 255]}, translate=(self.first_frame, 0, 0))
        labels_layer.contour=5

        # self.viewer.layers.selection.center()
        self.update_feature_plots()

        # Add vertical lines to plots
        for feat, pw in self.plots_widget.plot_widgets.items():
            vline = pg.InfiniteLine(pos=self.first_frame, angle=90, pen=pg.mkPen('r', width=2))
            pw.addItem(vline)
            self.vlines[feat] = vline
            if not np.isnan(self.cell_death):
                vline_death = pg.InfiniteLine(pos=self.cell_death, angle=90, pen=pg.mkPen('k', width=2))
                pw.addItem(vline_death)

    def update_feature_plots(self):
        for feat in self.feature_names:
            y_data = self.hdf5_file['Cells']['Phase'][feat][self.first_frame:self.last_frame, self.cell_idx]
            x_data = np.arange(self.first_frame, self.last_frame)
            pw = self.plots_widget.plot_widgets[feat]
            pw.getPlotItem().clear()
            pw.plot(x_data, y_data, pen=pg.mkPen('k'))

    def update_vertical_lines(self, event):
        current_frame = self.viewer.dims.current_step[0]
        # actual_frame = self.first_frame + current_frame
        for vline in self.vlines.values():
            vline.setPos(current_frame)
            
    def closeEvent(self, event):
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            self.loader_thread.quit()  # ask Qt thread to stop
            self.loader_thread.wait()  # block until it finishes
        event.accept()
            
class CellLoaderThread(QThread):
    start_signal = Signal(int)
    progress = Signal(int)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray)  # phase_data, epi_data, cell_outline

    def __init__(self, cell_idx, first_frame, last_frame, frame_size, hdf5_file):
        super().__init__()
        self.cell_idx = cell_idx
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.frame_size = frame_size
        self.hdf5_file = hdf5_file
        
        self.loading_bar = None
        

    def run(self):
        try:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_frames = self.last_frame - self.first_frame
            self.start_signal.emit(self.n_frames)
            
            # self.loading_bar = LoadingBarDialog(n_frames, f'Loading Cell {self.cell_idx}')
            # self.loading_bar.show()
            im_shape = self.hdf5_file['Images']['Phase'][0].shape
            phase_data = np.empty((self.n_frames, self.frame_size, self.frame_size))
            epi_data = np.empty((self.n_frames, self.frame_size, self.frame_size))
            mask = np.empty((self.n_frames, self.frame_size, self.frame_size), dtype=np.int32)

            # with h5py.File(self.hdf5_file, 'r') as f:
            x_centres = self.hdf5_file['Cells']['Phase']['X'][self.first_frame:self.last_frame, self.cell_idx]
            y_centres = self.hdf5_file['Cells']['Phase']['Y'][self.first_frame:self.last_frame, self.cell_idx]
            x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

            for idx in range(self.n_frames):
                frame_idx = self.first_frame + idx
                xmin, xmax, ymin, ymax = mask_funcs.get_crop_indices(
                    (y_centres[idx], x_centres[idx]), self.frame_size, im_shape
                )
                phase_data[idx] =  self.hdf5_file['Images']['Phase'][frame_idx, ymin:ymax, xmin:xmax]
                epi_data[idx] = self.hdf5_file['Images']['Epi'][frame_idx, ymin:ymax, xmin:xmax]
                mask[idx] = self.hdf5_file['Segmentations']['Phase'][frame_idx, ymin:ymax, xmin:xmax]
                # self.loading_bar.update_progress(idx)
                self.progress.emit(idx)

            cell_mask = (mask == self.cell_idx)
            # print(np.unique(cell_mask))

            self.finished.emit(phase_data, epi_data, cell_mask)
            # self.loading_bar.close()
            

        except Exception as e:
            import traceback
            traceback.print_exc()
