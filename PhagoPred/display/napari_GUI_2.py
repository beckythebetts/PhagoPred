from pathlib import Path

import dask.array as da
import h5py
import napari

from qtpy import QtWidgets
import h5py

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
        self._show_ims()
        
        self.status_label = QtWidgets.QLabel(" ")
        self.viewer.window.add_dock_widget(self.status_label, area='bottom')
        
        self._connect_signals()
        
    def _connect_signals(self):
        self.viewer.window.qt_viewer.destroyed.connect(self._close_hdf5)
        self._connect_mouse_interactions()
        
    def _load_data(self):

        # Suppose your dataset is stored at f["images"]
        hdf5_phase = self.hdf5_file['Images']['Phase']
        hdf5_epi = self.hdf5_file['Images']['Epi']
        hdf5_seg = self.hdf5_file['Segmentations']['Phase']

        # Wrap it in a Dask array (each chunk corresponds to HDF5 chunks)
        self.phase_data = da.from_array(hdf5_phase, chunks=hdf5_phase.chunks)
        self.epi_data = da.from_array(hdf5_epi, chunks=hdf5_epi.chunks)
        
        self.seg_data = da.from_array(hdf5_seg, chunks=hdf5_seg.chunks)
        
    def _show_ims(self):
        self.viewer.add_image(self.phase_data, name='Phase', colormap='gray')
        self.viewer.add_image(self.epi_data, name = 'Epi', colormap='red')
        
        self.labels_layer = self.viewer.add_labels(self.seg_data + 1, name="Semgentations")
    
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
                print(f"Clicked cell {label_val}")
            return True  # Stop further processing
        
        self.viewer.mouse_move_callbacks.append(_hover_callback)
        self.viewer.mouse_double_click_callbacks.append(_click_callback)
    
def main():
    hdf5_file = Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/21_10_2500.h5')
    viewer = napari.Viewer()
    all_cells = AllCellsViewer(viewer, hdf5_file)
    napari.run()
    
if __name__ == '__main__':
    main()