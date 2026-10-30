"""GUI for manually labelling death frames of randomly sampled cells.

Labels are saved to deaths.json in the dataset directory, in the same format
used by population_analysis (h5 file name -> {cell idx: death frame}, with
null for cells which stay alive). Skipped cells are saved separately to
deaths_skipped.json so they aren't sampled again but don't count as alive.

If label_priority.csv (from score_unlabelled.py) is in the dataset
directory, most cells are taken in its order, most uncertain for the model
first, and the rest sampled at random. Targeted cells are recorded in
deaths_targeted.json, as they're harder than average and so bias metrics
and alive counts computed from labels; random cells don't.

Usage:
    python -m PhagoPred.death_classifier.label_gui ~/thor_server/MacrophageData/02_10
"""
from pathlib import Path
from typing import Optional
import argparse
import json
import random

import h5py
import napari
import pandas as pd
from qtpy import QtWidgets

from PhagoPred.death_classifier.dataset import unlabelled_cells
from PhagoPred.display.GUI.one_cell import CellViewer
from PhagoPred.utils.h5_file import H5File


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def _save_json(data: dict, path: Path) -> None:
    """Write via a temporary file so a crash can't corrupt existing labels."""
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    tmp_path.replace(path)


class DeathLabeller(QtWidgets.QWidget):
    """Show random unlabelled cells and record their death frames."""

    def __init__(self,
                 viewer: napari.Viewer,
                 h5_files: list[Path],
                 labels_path: Path,
                 min_track_length: int = 50,
                 seed: Optional[int] = None,
                 priority_path: Optional[Path] = None,
                 random_frac: float = 0.2):
        """
        Args:
            viewer: napari viewer to display cells in.
            h5_files: datasets to sample cells from.
            labels_path: json to save death frames to.
            min_track_length: only sample cells tracked for at least this many
                frames (short tracks are mostly debris).
            seed: random seed for sampling cells.
            priority_path: csv of cells (file, cell columns) in the order to
                label them, from score_unlabelled.py. If None or missing,
                all cells are sampled at random.
            random_frac: with a priority list, fraction of cells still
                sampled at random, so the labels keep an unbiased sample.
        """
        super().__init__()
        self.viewer = viewer
        self.labels_path = labels_path
        self.skipped_path = labels_path.with_name(
            f'{labels_path.stem}_skipped.json')
        self.targeted_path = labels_path.with_name(
            f'{labels_path.stem}_targeted.json')
        self.labels = _load_json(self.labels_path)
        self.skipped = _load_json(self.skipped_path)
        self.targeted = _load_json(self.targeted_path)
        self.files = {path.name: h5py.File(path, 'r') for path in h5_files}
        # For CellViewer; unfiltered, so bad tracks can be shown and labelled
        self.h5s = {path.name: H5File(path) for path in h5_files}
        for h5 in self.h5s.values():
            h5.bad_track_thresh = None
        self.rng = random.Random(seed)
        self.random_frac = random_frac

        self.candidates = [(name, cell) for name, f in self.files.items()
                           for cell in unlabelled_cells(
                               f, name, self.labels, self.skipped,
                               min_track_length)]
        self.priority = []
        if priority_path is not None and priority_path.exists():
            remaining = set(self.candidates)
            self.priority = [
                (name, int(cell)) for name, cell in pd.read_csv(priority_path)
                [['file', 'cell']].itertuples(index=False)
                if (name, int(cell)) in remaining
            ]
        self.current = None
        self.current_targeted = False
        self.cell_viewer = None

        self._build_ui()
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._close)
        self._next_cell()

    def _pick_cell(self) -> tuple[tuple[str, int], bool]:
        """Next (file, cell) to label, and whether it came from the priority
        list rather than random sampling."""
        if self.priority and self.rng.random() >= self.random_frac:
            cell = self.priority.pop(0)
            self.candidates.remove(cell)
            return cell, True
        cell = self.candidates.pop(self.rng.randrange(len(self.candidates)))
        if cell in self.priority:
            self.priority.remove(cell)
        return cell, False

    def _record_targeted(self) -> None:
        if not self.current_targeted:
            return
        name, cell = self.current
        self.targeted = _load_json(self.targeted_path)
        self.targeted.setdefault(name, []).append(cell)
        _save_json(self.targeted, self.targeted_path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Open a specific cell, e.g. to check or correct its label
        open_row = QtWidgets.QHBoxLayout()
        self.file_select = QtWidgets.QComboBox()
        self.file_select.addItems(sorted(self.files))
        open_row.addWidget(self.file_select)
        open_row.addWidget(QtWidgets.QLabel('cell'))
        self.cell_input = QtWidgets.QLineEdit()
        self.cell_input.setFixedWidth(60)
        self.cell_input.returnPressed.connect(self._open_cell)
        open_row.addWidget(self.cell_input)
        open_button = QtWidgets.QPushButton('Open')
        open_button.clicked.connect(self._open_cell)
        open_row.addWidget(open_button)
        layout.addLayout(open_row)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        frame_row = QtWidgets.QHBoxLayout()
        frame_row.addWidget(QtWidgets.QLabel('Death frame:'))
        self.frame_input = QtWidgets.QLineEdit()
        self.frame_input.setFixedWidth(60)
        self.frame_input.returnPressed.connect(self._save_death)
        frame_row.addWidget(self.frame_input)
        self.current_frame_button = QtWidgets.QPushButton('Use current frame')
        # dims.point is the frame in world coordinates; dims.current_step is
        # an index relative to the start of the layers, which begin at the
        # cell's first frame
        self.current_frame_button.clicked.connect(
            lambda: self.frame_input.setText(
                str(int(round(self.viewer.dims.point[0])))))
        frame_row.addWidget(self.current_frame_button)
        layout.addLayout(frame_row)

        self.dead_button = QtWidgets.QPushButton('Save death frame')
        self.dead_button.clicked.connect(self._save_death)
        self.alive_button = QtWidgets.QPushButton('Alive (no death)')
        self.alive_button.clicked.connect(lambda: self._save_label(None))
        self.skip_button = QtWidgets.QPushButton('Skip (unsure / bad track)')
        self.skip_button.clicked.connect(self._skip)
        for button in (self.dead_button, self.alive_button, self.skip_button):
            layout.addWidget(button)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.frame_input, self.current_frame_button,
                       self.dead_button, self.alive_button, self.skip_button):
            widget.setEnabled(enabled)

    def _summary(self) -> str:
        all_labels = [
            v for cells in self.labels.values() for v in cells.values()
        ]
        num_dead = sum(v is not None for v in all_labels)
        summary = (f'Labelled {len(all_labels)} cells '
                   f'({num_dead} dead, {len(all_labels) - num_dead} alive), '
                   f'{len(self.candidates)} left to sample')
        if self.priority:
            summary += f' ({len(self.priority)} in priority list)'
        return summary

    def _next_cell(self):
        if not self.candidates:
            self._clear_cell()
            self.info_label.setText('No unlabelled cells left.')
            self._set_enabled(False)
            return
        # Whether the cell is targeted isn't shown, so it can't influence
        # the label
        self._show_cell(*self._pick_cell())

    def _open_cell(self):
        name = self.file_select.currentText()
        text = self.cell_input.text().strip()
        num_cells = self.files[name]['Cells']['Phase']['First Frame'].shape[1]
        if not text.isdigit() or int(text) >= num_cells:
            self.status_label.setText(
                f'Cell must be an integer from 0 to {num_cells - 1}')
            return
        self._show_cell((name, int(text)), False, opened=True)

    def _existing_label(self, name: str, cell: int) -> str:
        labels = _load_json(self.labels_path).get(name, {})
        if str(cell) in labels:
            frame = labels[str(cell)]
            return 'alive' if frame is None else f'died at {frame}'
        if cell in _load_json(self.skipped_path).get(name, []):
            return 'skipped (unsure / bad track)'
        return 'unlabelled'

    def _show_cell(self,
                   cell: tuple[str, int],
                   targeted: bool,
                   opened: bool = False) -> None:
        """Display a cell for labelling; opened for a cell chosen by the user
        rather than sampled."""
        self._clear_cell()
        self.frame_input.clear()
        self.current, self.current_targeted = cell, targeted
        name, cell = cell
        self.cell_viewer = CellViewer(self.viewer, self.h5s[name], cell)
        existing = ''
        if opened:
            # Show its current label, which saving replaces
            existing = f'Current label: {self._existing_label(name, cell)}<br>'
            frame = _load_json(self.labels_path).get(name, {}).get(str(cell))
            if frame is not None:
                self.frame_input.setText(str(frame))
        self.info_label.setText(
            f'<b>{name}, cell {cell}</b><br>'
            f'Frames {self.cell_viewer.first_frame}-'
            f'{self.cell_viewer.last_frame}<br>{existing}{self._summary()}')

        # Wait for the crops to load before allowing labelling
        self._set_enabled(False)
        self.cell_viewer.loader_thread.finished.connect(
            lambda *_: self._set_enabled(True))

    def _clear_cell(self):
        if self.cell_viewer is None:
            return
        self.viewer.dims.events.current_step.disconnect(
            self.cell_viewer.update_vertical_lines)
        if self.cell_viewer.loader_thread.isRunning():
            self.cell_viewer.loader_thread.wait()
        self.viewer.window.remove_dock_widget(self.cell_viewer.plots_dock)
        self.viewer.layers.clear()
        self.cell_viewer = None

    def _save_death(self):
        text = self.frame_input.text().strip()
        name, _ = self.current
        num_frames = self.files[name]['Images']['Phase'].shape[0]
        if not text.isdigit() or int(text) >= num_frames:
            self.status_label.setText(
                f'Death frame must be an integer from 0 to {num_frames - 1}')
            return
        self._save_label(int(text))

    def _save_label(self, death_frame: Optional[int]):
        name, cell = self.current
        # Re-read first, so edits made to the file while the GUI is open
        # (e.g. corrected labels) aren't overwritten
        self.labels = _load_json(self.labels_path)
        self.labels.setdefault(name, {})[str(cell)] = death_frame
        _save_json(self.labels, self.labels_path)
        # A cell is either labelled or skipped (skipped cells are treated as
        # bad tracks, overriding any label), so relabelling un-skips it
        self.skipped = _load_json(self.skipped_path)
        if cell in self.skipped.get(name, []):
            self.skipped[name].remove(cell)
            _save_json(self.skipped, self.skipped_path)
        self._record_targeted()
        self._forget_current()
        self.status_label.setText(
            f'Saved {name} cell {cell}: '
            f'{"alive" if death_frame is None else f"died at {death_frame}"}')
        self._next_cell()

    def _skip(self):
        name, cell = self.current
        self.skipped = _load_json(self.skipped_path)
        if cell not in self.skipped.get(name, []):
            self.skipped.setdefault(name, []).append(cell)
            _save_json(self.skipped, self.skipped_path)
        self.labels = _load_json(self.labels_path)
        if str(cell) in self.labels.get(name, {}):
            del self.labels[name][str(cell)]
            _save_json(self.labels, self.labels_path)
        self._record_targeted()
        self._forget_current()
        self.status_label.setText(f'Skipped {name} cell {cell}')
        self._next_cell()

    def _forget_current(self) -> None:
        """Stop sampling the current cell, e.g. one opened directly."""
        if self.current in self.candidates:
            self.candidates.remove(self.current)
        if self.current in self.priority:
            self.priority.remove(self.current)

    def _close(self):
        for f in self.files.values():
            f.close()


def run(dataset_dir: Path,
        labels_name: str = 'deaths.json',
        min_track_length: int = 50,
        seed: Optional[int] = None,
        priority_name: str = 'label_priority.csv',
        random_frac: float = 0.2) -> None:
    """Label cells from all h5 files in dataset_dir.

    Cells are taken in order from priority_name in dataset_dir if it exists
    (with random_frac sampled at random), otherwise all at random.
    """
    h5_files = sorted(dataset_dir.glob('*.h5'))
    if not h5_files:
        raise FileNotFoundError(f'No .h5 files found in {dataset_dir}')

    viewer = napari.Viewer()
    qt_window = viewer.window._qt_window
    qt_window.resize(1200, 800)

    labeller = DeathLabeller(viewer, h5_files, dataset_dir / labels_name,
                             min_track_length, seed,
                             dataset_dir / priority_name, random_frac)
    viewer.window.add_dock_widget(labeller, area='right', name='Label death')
    napari.run()


if __name__ == '__main__':
    # run(Path('~/thor_server').expanduser() / 'MacrophageData' / '02_10')
    run(Path('temp') / 'death_labelling_02_10')
    # parser = argparse.ArgumentParser(
    #     description='Label death frames of randomly sampled cells.')
    # parser.add_argument('dataset_dir',
    #                     type=Path,
    #                     help='Directory containing the .h5 files.')
    # parser.add_argument('--labels',
    #                     default='deaths.json',
    #                     help='Labels file name, saved in dataset_dir.')
    # parser.add_argument('--min-track-length', type=int, default=50)
    # parser.add_argument('--seed', type=int, default=None)
    # args = parser.parse_args()
    # run(args.dataset_dir.expanduser(), args.labels, args.min_track_length,
    #     args.seed)
