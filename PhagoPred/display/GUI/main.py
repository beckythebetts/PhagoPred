import napari
from pathlib import Path

from PhagoPred.display.GUI.all_cells import AllCellsViewer


def main():
    hdf5_file = Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/28_10_2500.h5')
    viewer = napari.Viewer()
    qt_window = viewer.window._qt_window  # access the underlying QMainWindow
    qt_window.resize(1200, 800)  # or whatever fits your VM screen

# Optionally, set minimum and maximum sizes so the user can still resize freely
    qt_window.setMinimumSize(800, 600)
    qt_window.setMaximumSize(1920, 1080)
    
    all_cells = AllCellsViewer(viewer, hdf5_file)
    napari.run()
    
if __name__ == '__main__':
    main()