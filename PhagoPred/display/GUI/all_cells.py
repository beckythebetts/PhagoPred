from pathlib import Path

import dask.array as da
import h5py
import napari
from qtpy import QtWidgets
import numpy as np
from tqdm import tqdm
import numba

from PhagoPred.display.GUI.one_cell import CellViewer

class AllCellsViewer:
    """Lazily load all ims from hdf5 file."""
    def __init__(self, viewer, hdf5_file_path: Path):
        self.viewer = viewer
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(self.hdf5_file_path, mode="r")
        
        self.phase_data = None
        self.epi_data = None
        self.seg_data = None
        
        self._load_data()
        # self._fill_missing_segs()
        self._show_ims()
        
        self.status_label = QtWidgets.QLabel(" ")
        self.cell_death_label = QtWidgets.QLabel("")
        label_dock = self.viewer.window.add_dock_widget(self.status_label, area='bottom')
        self.viewer.window.add_dock_widget(self.cell_death_label, area='bottom')
        
        self.back_button = QtWidgets.QPushButton("← Back to All Cells")
        self.back_button.setVisible(False)
        self.viewer.window.add_dock_widget(self.back_button, area="top")    
        
        self._connect_signals()
        
    def _connect_signals(self):
        # self.viewer.events.destroyed.connect(self._close_hdf5)
        app = QtWidgets.QApplication.instance()
        app.aboutToQuit.connect(self._close_hdf5)
        self._connect_mouse_interactions()
        self.back_button.clicked.connect(self._return_to_overview)
        
    def _load_data(self):

        # Suppose your dataset is stored at f["images"]
        hdf5_phase = self.hdf5_file['Images']['Phase']
        hdf5_epi = self.hdf5_file['Images']['Epi']
        hdf5_seg = self.hdf5_file['Segmentations']['Phase']

        # Wrap it in a Dask array (each chunk corresponds to HDF5 chunks)
        self.phase_data = da.from_array(hdf5_phase, chunks=hdf5_phase.chunks)
        self.epi_data = da.from_array(hdf5_epi, chunks=hdf5_epi.chunks)
        
        self.seg_data = da.from_array(hdf5_seg, chunks=hdf5_seg.chunks)
    
    def _fill_missing_segs(self):
        first_appearances = self.hdf5_file['Cells']['Phase']['First Frame'][0]
        last_appearances = self.hdf5_file['Cells']['Phase']['Last Frame'][0]
        self.seg_data = fill_missing_cells(self.seg_data, first_appearances, last_appearances)
            
        
    def _show_ims(self):
        if "Phase" not in self.viewer.layers:
            self.viewer.add_image(self.phase_data, name='Phase', colormap='gray', opacity=1.0)
            self.viewer.add_image(self.epi_data, name='Epi', colormap='red', blending='additive', opacity=1.0)
            self.labels_layer = self.viewer.add_labels(self.seg_data + 1, name="Segmentations", opacity=0.3)
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
    
    def _close_hdf5(self):
        self.hdf5_file.close()
        
    def _connect_mouse_interactions(self):
        """Update status bar with label ID under cursor."""
        def _hover_callback(viewer, event):
            label_val = self.labels_layer.get_value(event.position)
            if label_val:
                if label_val > 0:
                    self.status_label.setText(f'Cell index: {label_val-1}')
                    return True  
            self.status_label.setText('')
            return True
            
        def _click_callback(viewer, event):
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
        self.cell_viewer = CellViewer(self.viewer, self.hdf5_file, cell_idx)
        self.cell_viewer.cell_death_signal.connect(lambda x: self.cell_death_label.setText(f'Death at frame: {x}'))
        
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
                        filled_mask = np.where(reused_mask, cell_id, filled_mask)
            else:
                # store last known mask
                last_known_masks[cell_id] = (mask == cell_id)

        filled_frames.append(filled_mask)

    # Stack filled frames back into a Dask array
    # filled_seg_data = np.stack()
    filled_seg_data = da.stack([da.from_array(frame, chunks=seg_data.chunksize[1:]) 
                                for frame in filled_frames], axis=0)
    return filled_seg_data

