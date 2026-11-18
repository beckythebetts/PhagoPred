from __future__ import annotations
from functools import lru_cache, partial

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

# Recently shown frames kept in memory, so scrubbing back over them doesn't
# re-read the (remote) file
FRAME_CACHE_SIZE = 32


def _alive_seg_reader(h5: H5File) -> Callable[[int], np.ndarray]:
    """get_frame(i) for segmentations with bad tracks and dead cells hidden
    (as get_segs(filter_bad=True, alive_only=True)), reading the per-cell
    tables once up front instead of on every frame. Either table may be
    missing (a freshly segmented dataset has neither): that filter is then
    skipped, and with neither the frames are shown as they are."""
    # Own instance: h5 itself has its bad-track filter turned off
    filtered = H5File(h5.path)
    good = (filtered.bad_track_prob <= filtered.bad_track_thresh
            if filtered.has_feature('Bad Track Probability') else None)
    death = (filtered.death_frames
             if filtered.has_feature('Death Frame') else None)
    if good is None and death is None:
        return h5.get_segs

    num_cells = len(good if good is not None else death)

    def get_frame(i: int) -> np.ndarray:
        segs = h5.get_segs(i)
        # (cells + 1,): last entry is background (-1), never hidden
        hide = np.zeros(num_cells + 1, dtype=bool)
        if good is not None:
            hide[:-1] |= ~good
        if death is not None:
            hide[:-1] |= death <= i  # NaN death compares False
        return np.where(hide[segs], -1, segs).astype(segs.dtype)

    return get_frame


def _lazy_stack(get_frame: Callable[[int], np.ndarray], shape: tuple,
                dtype: np.dtype) -> da.Array:
    """(T, H, W) dask array reading each frame with get_frame only when
    napari displays it, so no file is held open."""
    get_frame = lru_cache(maxsize=FRAME_CACHE_SIZE)(get_frame)
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
        self.missed_layer = None
        self.missed_points = np.empty((0, 3))  # (frame, y, x)
        self.removed_layer = None
        # Segmentations left out of the count: (frame, label) -> (frame, y, x)
        # of where it was clicked, to mark it. Display/counting only.
        self.removed = {}
        self.count_label = QtWidgets.QLabel("")
        self.cell_info = None

        self._load_data()
        # self._fill_missing_segs()
        self._show_ims()

        self.status_label = QtWidgets.QLabel(" ")
        self.cell_death_label = QtWidgets.QLabel("")
        self.cell_event_label = QtWidgets.QLabel("")
        label_dock = self.viewer.window.add_dock_widget(self.status_label,
                                                        area='right')
        self.viewer.window.add_dock_widget(self.cell_death_label, area='right')
        self.viewer.window.add_dock_widget(self.cell_event_label, area='right')

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

        counter_widget = QtWidgets.QWidget()
        counter_layout = QtWidgets.QHBoxLayout(counter_widget)
        counter_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_frame_button = QtWidgets.QPushButton("Reset frame")
        self.reset_all_button = QtWidgets.QPushButton("Reset all")
        counter_layout.addWidget(self.count_label)
        counter_layout.addWidget(self.reset_frame_button)
        counter_layout.addWidget(self.reset_all_button)
        self.viewer.window.add_dock_widget(counter_widget, area="right")

        count_actions = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(count_actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_button = QtWidgets.QPushButton("Remove segmentation")
        self.remove_button.setCheckable(True)
        self.remove_button.setToolTip(
            "While on, double-click a segmentation to leave it out of the "
            "count (the data is not changed); double-click again to restore.")
        self.save_count_button = QtWidgets.QPushButton("Save count")
        self.save_count_button.setToolTip(
            "Save this frame's corrected count in the .h5 file's attributes "
            "as num_cells_<frame>.")
        actions_layout.addWidget(self.remove_button)
        actions_layout.addWidget(self.save_count_button)
        self.viewer.window.add_dock_widget(count_actions, area="right")
        self._update_count()

        self._connect_signals()

    def _connect_signals(self):
        self._connect_mouse_interactions()
        self.back_button.clicked.connect(self._return_to_overview)
        self.goto_button.clicked.connect(self._goto_cell_from_text)
        self.goto_input.returnPressed.connect(self._goto_cell_from_text)
        self.reset_frame_button.clicked.connect(
            partial(self._reset_missed, frame_only=True))
        self.reset_all_button.clicked.connect(
            partial(self._reset_missed, frame_only=False))
        self.viewer.dims.events.current_step.connect(self._update_count)
        self.remove_button.toggled.connect(self._on_remove_mode)
        self.save_count_button.clicked.connect(self._save_count)

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
            self.epi_data = _lazy_stack(
                partial(self.h5.get_epi, threshold=True), *epi_info)
            self.epi_contrast = epi_contrast_limits(self.h5)

        seg_info = self.h5.ims_info(H5File.SEGS)
        if seg_info is not None:
            self.seg_data = _lazy_stack(_alive_seg_reader(self.h5), *seg_info)

        self.cell_info = {}
        for name, feature in [('Death', 'Death Frame'),
                              ('Event', 'EventDetection'),
                              ('Clearance', 'Clearance')]:
            try:
                self.cell_info[name] = self.h5.get_cell_feature(feature)
            except KeyError:
                pass

    def _fill_missing_segs(self):
        self.seg_data = fill_missing_cells(self.seg_data, self.h5.first_frames,
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
            self._add_missed_layer()
            self._add_removed_layer()
        else:
            for layer_name in ("Phase", "Epi", "Segmentations", "Missed cells",
                               "Removed segmentations"):
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = True

    def _add_missed_layer(self):
        """Points layer for clicking on cells the segmentation missed; kept
        in missed_points so they survive opening a cell."""
        self.missed_layer = self.viewer.add_points(self.missed_points,
                                                   ndim=3,
                                                   name='Missed cells',
                                                   face_color='cyan',
                                                   size=15)
        self.missed_layer.mode = 'add'
        self.missed_layer.events.data.connect(self._on_missed_edit)

    def _on_missed_edit(self, event=None):
        self.missed_points = self.missed_layer.data
        self._update_count()

    def _add_removed_layer(self):
        """Marks segmentations left out of the count. Read-only: edited by
        double-clicking segmentations in remove mode."""
        self.removed_layer = self.viewer.add_points(
            self._removed_points(),
            ndim=3,
            name='Removed segmentations',
            face_color='transparent',
            border_color='red',
            border_width=0.15,
            symbol='cross',
            size=25)
        self.removed_layer.editable = False

    def _removed_points(self) -> np.ndarray:
        return (np.array(list(self.removed.values()), dtype=float).reshape(
            -1, 3))

    def _on_remove_mode(self, checked: bool):
        """In remove mode double-clicks remove segmentations, so stop
        clicks adding missed cells."""
        if self.missed_layer is not None:
            self.missed_layer.mode = 'pan_zoom' if checked else 'add'

    def _toggle_removed(self, position, label_val: int):
        key = (int(round(position[0])), int(label_val))
        if key in self.removed:
            del self.removed[key]
        else:
            self.removed[key] = tuple(position[:3])
        if self.removed_layer in self.viewer.layers:
            self.removed_layer.data = self._removed_points()
        self._update_count()

    def _frame_counts(self, frame: int) -> dict | None:
        """Counts for a frame, or None if it can't be worked out (no
        segmentations, or not in the overview)."""
        if (self.labels_layer is None or not self.labels_layer.visible
                or not 0 <= frame < len(self.labels_layer.data)):
            return None
        labels = np.unique(np.asarray(self.labels_layer.data[frame]))
        labels = labels[labels > 0]
        removed = sum(1 for f, label in self.removed
                      if f == frame and label in labels)
        missed = int(np.sum(np.round(self.missed_points[:, 0]) == frame))
        segmented = len(labels)
        return dict(segmented=segmented,
                    removed=removed,
                    missed=missed,
                    total=segmented - removed + missed)

    def _update_count(self, event=None):
        frame = int(self.viewer.dims.current_step[0])
        on_frame = np.round(self.missed_points[:, 0]) == frame
        text = (f'Missed cells: {on_frame.sum()} this frame, '
                f'{len(self.missed_points)} total')
        counts = self._frame_counts(frame)
        if counts is not None:
            text += (f"\nSegmented {counts['segmented']} - removed "
                     f"{counts['removed']} + missed {counts['missed']} = "
                     f"{counts['total']}")
        self.count_label.setText(text)

    def _save_count(self):
        """Write this frame's corrected count into the file's attributes."""
        frame = int(self.viewer.dims.current_step[0])
        counts = self._frame_counts(frame)
        if counts is None:
            self.status_label.setText('No segmentations to count')
            return
        name = f'num_cells_{frame}'
        try:
            self.h5.write_attr(name, counts['total'])
        except OSError as e:
            self.status_label.setText(f'Could not save {name}: {e}')
            return
        self.status_label.setText(f"Saved {name} = {counts['total']}")

    def _reset_missed(self, frame_only: bool):
        """Clear missed cells and removed segmentations, on this frame or
        everywhere."""
        frame = int(self.viewer.dims.current_step[0])
        if frame_only:
            keep = np.round(self.missed_points[:, 0]) != frame
            self.removed = {k: v for k, v in self.removed.items()
                            if k[0] != frame}
        else:
            keep = np.zeros(len(self.missed_points), dtype=bool)
            self.removed = {}
        self.missed_points = self.missed_points[keep]
        if self.missed_layer in self.viewer.layers:
            self.missed_layer.data = self.missed_points
        if self.removed_layer in self.viewer.layers:
            self.removed_layer.data = self._removed_points()
        self._update_count()

    def _hide_overview_layers(self):
        """Hide the overview (full field) layers."""
        for layer_name in ("Phase", "Epi", "Segmentations", "Missed cells",
                           "Removed segmentations"):
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

    def _return_to_overview(self):
        """Return to full-field view."""
        # Remove cell-specific layers
        if self.cell_viewer is not None:
            self.viewer.layers.clear()
            if self.cell_viewer.plots_dock is not None:
                self.cell_viewer.plots_dock.close()
            self.cell_viewer = None

        # Show the overview layers again
        self._show_ims()
        self.back_button.setVisible(False)
        self.cell_death_label.setText("")

    def _hover_text(self, cell_idx: int) -> str:
        lines = [f'Cell index: {cell_idx}']
        if self.cell_info is not None:
            for name, frames in self.cell_info.items():
                if cell_idx < len(frames) and not np.isnan(frames[cell_idx]):
                    lines.append(f'{name} frame: {int(frames[cell_idx])}')
        return '\n'.join(lines)

    def _connect_mouse_interactions(self):
        """Update status bar with label ID under cursor."""

        def _hover_callback(viewer, event):
            if self.labels_layer is None:
                return True
            label_val = self.labels_layer.get_value(event.position)
            if label_val:
                if label_val > 0:
                    self.status_label.setText(self._hover_text(label_val - 1))
                    return True
            self.status_label.setText('')
            return True

        def _click_callback(viewer, event):
            if self.labels_layer is None:
                return True
            if self.remove_button.isChecked():
                label_val = self.labels_layer.get_value(event.position)
                if label_val:
                    self._toggle_removed(event.position, label_val)
                return True
            if (viewer.layers.selection.active is self.missed_layer
                    and self.missed_layer.mode == 'add'):
                return True  # counting: don't open the cell
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
        self.cell_viewer.cell_event_signal.connect(
            lambda x: self.cell_event_label.setText(
                'No event' if x is None else f'Event at frame: {x}'))


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
