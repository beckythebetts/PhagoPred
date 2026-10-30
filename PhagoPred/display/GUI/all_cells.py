from __future__ import annotations

from pathlib import Path
from typing import Callable

import dask
import dask.array as da
import napari
from qtpy import QtWidgets
import numpy as np
from tqdm import tqdm

from PhagoPred.display.GUI.one_cell import CellViewer
from PhagoPred.utils.h5_file import H5File


def _lazy_stack(get_frame: Callable[[int], np.ndarray], shape: tuple,
                dtype: np.dtype) -> da.Array:
    """(T, H, W) dask array reading each frame with get_frame only when
    napari displays it, so no file is held open."""
    frames = [
        da.from_delayed(dask.delayed(get_frame)(i),
                        shape=shape[1:],
                        dtype=dtype) for i in range(shape[0])
    ]
    return da.stack(frames)


def epi_contrast_limits(h5: H5File,
                        num_samples: int = 3,
                        upper_percentile: float = 99.9) -> tuple:
    """Display limits for Epi from a few frames spread over the movie.

    napari's automatic limits come from a single frame, which for raw 16
    bit Epi is often all black or saturated: fluorescence grows over the
    movie, so later frames dominate the upper limit here.

    Returns:
        (contrast_limits, contrast_limits_range): limits to start with, and
        the slider range (0 to the brightest sampled pixel).
    """
    frames = np.linspace(0, h5.num_frames - 1, num_samples).astype(int)
    samples = h5.get_epi(frames)
    upper = float(np.percentile(samples, upper_percentile))
    brightest = float(samples.max())
    return (0.0, max(upper, 1.0)), (0.0, max(brightest, 1.0))


class AllCellsViewer:
    """Lazily load all ims from hdf5 file."""

    def __init__(self, viewer, hdf5_file_path: Path):
        self.viewer = viewer
        self.h5 = H5File(hdf5_file_path)
        # Every cell can be opened, including likely bad tracks
        self.h5.bad_track_thresh = None

        self.phase_data = None
        self.epi_data = None
        self.epi_contrast = None
        self.seg_data = None
        self.labels_layer = None
        self.cell_viewer = None

        self._load_data()
        # self._fill_missing_segs()
        self._show_ims()

        self.status_label = QtWidgets.QLabel(" ")
        self.cell_death_label = QtWidgets.QLabel("")
        label_dock = self.viewer.window.add_dock_widget(self.status_label,
                                                        area='right')
        self.viewer.window.add_dock_widget(self.cell_death_label, area='right')

        self.back_button = QtWidgets.QPushButton("← Back to All Cells")
        self.back_button.setVisible(False)
        self.viewer.window.add_dock_widget(self.back_button, area="right")

        self.goto_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.goto_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.goto_label = QtWidgets.QLabel("Go to cell:")
        self.goto_input = QtWidgets.QLineEdit()
        self.goto_input.setFixedWidth(60)
        self.goto_button = QtWidgets.QPushButton("Open")

        layout.addWidget(self.goto_label)
        layout.addWidget(self.goto_input)
        layout.addWidget(self.goto_button)

        self.viewer.window.add_dock_widget(self.goto_widget, area="right")

        self._connect_signals()

    def _connect_signals(self):
        self._connect_mouse_interactions()
        self.back_button.clicked.connect(self._return_to_overview)
        self.goto_button.clicked.connect(self._goto_cell_from_text)
        self.goto_input.returnPressed.connect(self._goto_cell_from_text)

    def _goto_cell_from_text(self):
        """Open a cell index typed into the text box."""
        text = self.goto_input.text().strip()
        if not text.isdigit():
            self.status_label.setText("Invalid cell index")
            return

        cell_idx = int(text)

        # max_cell = self.seg_data.max().compute() - 1
        # if cell_idx < 0 or cell_idx > max_cell:
        #     self.status_label.setText(f"Cell index must be between 0 and {max_cell}")
        #     return

        self._open_cell_view(cell_idx)

    def _load_data(self):
        phase_info = self.h5.ims_info(H5File.PHASE)
        self.phase_data = _lazy_stack(self.h5.get_phase, *phase_info)

        epi_info = self.h5.ims_info(H5File.FLUOR)
        if epi_info is not None:
            self.epi_data = _lazy_stack(self.h5.get_epi, *epi_info)
            self.epi_contrast = epi_contrast_limits(self.h5)

        seg_info = self.h5.ims_info(H5File.SEGS)
        if seg_info is not None:
            self.seg_data = _lazy_stack(self.h5.get_segs, *seg_info)

    def _fill_missing_segs(self):
        self.seg_data = fill_missing_cells(self.seg_data,
                                           self.h5.first_frames,
                                           self.h5.last_frames)

    def _show_ims(self):
        if "Phase" not in self.viewer.layers:
            self.viewer.add_image(self.phase_data,
                                  name='Phase',
                                  colormap='gray',
                                  contrast_limits=(0, 255),
                                  opacity=1.0)
            if self.epi_data is not None:
                limits, limits_range = self.epi_contrast
                epi_layer = self.viewer.add_image(self.epi_data,
                                                  name='Epi',
                                                  colormap='red',
                                                  blending='additive',
                                                  contrast_limits=limits,
                                                  opacity=1.0)
                epi_layer.contrast_limits_range = limits_range
            if self.seg_data is not None:
                self.labels_layer = self.viewer.add_labels(
                    self.seg_data + 1, name="Segmentations", opacity=0.3)
        else:
            for layer_name in ("Phase", "Epi", "Segmentations"):
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = True

    def _hide_overview_layers(self):
        """Hide the overview (full field) layers."""
        for layer_name in ("Phase", "Epi", "Segmentations"):
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

    def _return_to_overview(self):
        """Return to full-field view."""
        # Remove cell-specific layers
        if self.cell_viewer is not None:
            self.viewer.layers.clear()
            self.cell_viewer.plots_dock.close()
            self.cell_viewer = None

        # Show the overview layers again
        self._show_ims()
        self.back_button.setVisible(False)
        self.cell_death_label.setText("")

    def _connect_mouse_interactions(self):
        """Update status bar with label ID under cursor."""

        def _hover_callback(viewer, event):
            if self.labels_layer is None:
                return True
            label_val = self.labels_layer.get_value(event.position)
            if label_val:
                if label_val > 0:
                    self.status_label.setText(f'Cell index: {label_val-1}')
                    return True
            self.status_label.setText('')
            return True

        def _click_callback(viewer, event):
            if self.labels_layer is None:
                return True
            label_val = self.labels_layer.get_value(event.position)
            if label_val > 0:
                label_val = label_val - 1
                self._open_cell_view(label_val)
            return True  # Stop further processing

        self.viewer.mouse_move_callbacks.append(_hover_callback)
        self.viewer.mouse_double_click_callbacks.append(_click_callback)

    def _open_cell_view(self, cell_idx):
        self._hide_overview_layers()
        self.back_button.setVisible(True)

        # Create a CellViewer instance
        self.cell_viewer = CellViewer(self.viewer,
                                      self.h5,
                                      cell_idx,
                                      epi_contrast=self.epi_contrast)
        self.cell_viewer.cell_death_signal.connect(
            lambda x: self.cell_death_label.setText(
                'No death predicted' if x is None else f'Death at frame: {x}'))


def fill_missing_cells(seg_data, first_appearance, last_appearance):
    """
    seg_data: Dask array of shape (frames, height, width)
    num_cells: total number of cells
    first_appearance / last_appearance: lists of frames for each cell
    """
    last_known_masks = {}

    filled_frames = []

    num_cells = len(first_appearance)

    for i in tqdm(range(seg_data.shape[0])):
        # Compute the current frame to NumPy
        mask = seg_data[i].compute()  # pull frame into memory
        filled_mask = mask.copy()

        present_ids = np.unique(mask)
        present_ids = present_ids[present_ids > 0]  # ignore background

        for cell_id in range(num_cells):
            if cell_id not in present_ids:
                if first_appearance[cell_id] <= i <= last_appearance[cell_id]:
                    if cell_id in last_known_masks:
                        reused_mask = last_known_masks[cell_id]
                        filled_mask = np.where(reused_mask, cell_id,
                                               filled_mask)
            else:
                # store last known mask
                last_known_masks[cell_id] = (mask == cell_id)

        filled_frames.append(filled_mask)

    # Stack filled frames back into a Dask array
    # filled_seg_data = np.stack()
    filled_seg_data = da.stack([
        da.from_array(frame, chunks=seg_data.chunksize[1:])
        for frame in filled_frames
    ],
                               axis=0)
    return filled_seg_data
