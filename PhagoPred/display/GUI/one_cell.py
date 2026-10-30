from __future__ import annotations

import napari
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QThread, Signal

from PhagoPred.display.GUI.plots_widget import FeaturePlotsWidget
from PhagoPred.display.GUI.utils import LoadingBarDialog
from PhagoPred.utils import tools, mask_funcs
from PhagoPred.utils.h5_file import H5File

# Per-frame features not worth plotting
NOT_PLOTTED = ('X', 'Y', 'Macrophage', 'Dead Macrophage')


class CellViewer(QWidget):
    """Display a single cell with cropped images in Napari and feature plots."""

    # Death frame, or None if no death is predicted
    cell_death_signal = Signal(object)

    def __init__(self,
                 viewer: napari.Viewer,
                 h5: H5File,
                 cell_idx: int,
                 frame_size: int = 300,
                 epi_contrast: tuple | None = None):
        """
        Args:
            h5: the dataset; its bad_track_thresh should be None so bad
                tracks have frames to show.
            epi_contrast: (contrast_limits, contrast_limits_range) for the
                Epi crops, to match the overview; None lets napari choose.
        """
        super().__init__()

        self.viewer = viewer
        self.h5 = h5
        self.cell_idx = cell_idx
        self.frame_size = frame_size
        self.epi_contrast = epi_contrast

        self.phase_data = None
        self.epi_data = None

        self.first_frame = int(self.h5.first_frames[self.cell_idx])
        self.last_frame = int(self.h5.last_frames[self.cell_idx])
        death_frame = self.h5.death_frames[self.cell_idx]
        self.cell_death = None if np.isnan(death_frame) else int(death_frame)

        self.progress_bar = None
        self.feature_names = [
            n for n in self.h5.feature_names if n not in NOT_PLOTTED
        ]

        # Create plots widget with checkboxes
        self.plots_widget = FeaturePlotsWidget(self.feature_names)
        self.plots_dock = self.viewer.window.add_dock_widget(self.plots_widget,
                                                             area='bottom')
        self.plots_widget.setMinimumHeight(400)

        # Start loading images in thread
        self.loader_thread = CellLoaderThread(cell_idx=self.cell_idx,
                                              first_frame=self.first_frame,
                                              last_frame=self.last_frame,
                                              frame_size=self.frame_size,
                                              h5=self.h5)
        self.loader_thread.start_signal.connect(self._start_load)
        self.loader_thread.start()
        # Link Napari frame changes to vertical lines
        self.vlines = {}
        self.viewer.dims.events.current_step.connect(
            self.update_vertical_lines)

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

        self.viewer.add_image(self.phase_data,
                              name=f'Phase {self.cell_idx}',
                              colormap='gray',
                              contrast_limits=(0, 255),
                              translate=(self.first_frame, 0, 0))
        if self.epi_data is not None:
            epi_layer = self.viewer.add_image(
                self.epi_data,
                name=f'Epi {self.cell_idx}',
                blending='additive',
                colormap='red',
                opacity=0.5,
                translate=(self.first_frame, 0, 0))
            if self.epi_contrast is not None:
                limits, limits_range = self.epi_contrast
                epi_layer.contrast_limits_range = limits_range
                epi_layer.contrast_limits = limits
        labels_layer = self.viewer.add_labels(mask,
                                              name=f'Cell {self.cell_idx}',
                                              colormap={1: [255, 255, 0, 255]},
                                              translate=(self.first_frame, 0,
                                                         0))
        labels_layer.contour = 5

        # Centre the view on the cropped cell
        self.viewer.reset_view()

        self.update_feature_plots()

        # Add vertical lines to plots
        for feat, pw in self.plots_widget.plot_widgets.items():
            vline = pg.InfiniteLine(pos=self.first_frame,
                                    angle=90,
                                    pen=pg.mkPen('r', width=2))
            pw.addItem(vline)
            self.vlines[feat] = vline

    def update_feature_plots(self):
        x_data = np.arange(self.first_frame, self.last_frame)
        features = self.h5.get_features(self.feature_names,
                                        cell_idxs=self.cell_idx,
                                        frames=slice(self.first_frame,
                                                     self.last_frame))
        for feat, y_data in features.items():
            pw = self.plots_widget.plot_widgets[feat]
            pw.getPlotItem().clear()
            pw.plot(x_data, y_data, pen=pg.mkPen('k'))

    def update_vertical_lines(self, event):
        # current_step is relative to the start of the layers (first_frame),
        # dims.point is the actual frame
        current_frame = self.viewer.dims.point[0]
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
    # phase_data, epi_data (None if the file has no Epi), cell mask
    finished = Signal(np.ndarray, object, np.ndarray)

    def __init__(self, cell_idx: int, first_frame: int, last_frame: int,
                 frame_size: int, h5: H5File):
        super().__init__()
        self.cell_idx = cell_idx
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.frame_size = frame_size
        self.h5 = h5

    def run(self):
        try:
            self.n_frames = self.last_frame - self.first_frame
            self.start_signal.emit(self.n_frames)

            im_shape = self.h5.im_dims
            crop_shape = (self.n_frames, self.frame_size, self.frame_size)
            # Crops keep each stack's dtype (e.g. uint16 Epi)
            phase_data = np.empty(crop_shape,
                                  dtype=self.h5.ims_info(H5File.PHASE)[1])
            epi_info = self.h5.ims_info(H5File.FLUOR)
            epi_data = (None if epi_info is None else np.empty(
                crop_shape, dtype=epi_info[1]))
            mask = np.empty(crop_shape, dtype=np.int32)

            frames = slice(self.first_frame, self.last_frame)
            x_centres = tools.fill_nans(
                self.h5.get_features('X', self.cell_idx, frames))
            y_centres = tools.fill_nans(
                self.h5.get_features('Y', self.cell_idx, frames))

            for idx in range(self.n_frames):
                frame_idx = self.first_frame + idx
                # 'X' is the row centroid and 'Y' the column, so the centre
                # must be passed as (row, col) to match im_shape = (H, W)
                rmin, rmax, cmin, cmax = mask_funcs.get_crop_indices(
                    (x_centres[idx], y_centres[idx]), self.frame_size,
                    im_shape)
                region = (slice(rmin, rmax), slice(cmin, cmax))
                phase_data[idx] = self.h5.get_phase(frame_idx, region)
                if epi_data is not None:
                    epi_data[idx] = self.h5.get_epi(frame_idx, region)
                mask[idx] = self.h5.get_segs(frame_idx, region)
                self.progress.emit(idx)

            cell_mask = (mask == self.cell_idx)

            self.finished.emit(phase_data, epi_data, cell_mask)

        except Exception:
            import traceback
            traceback.print_exc()
