import argparse
import napari
from pathlib import Path

from PhagoPred.display.GUI.all_cells import AllCellsViewer
from PhagoPred import SETTINGS


def find_dataset(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path

    if not name.endswith('.h5'):
        name += '.h5'
        if Path(name).is_file():
            return Path(name)

    matches = list(Path.cwd().rglob(name))
    if not matches:
        raise FileNotFoundError(f"No dataset named '{name}' found under {Path.cwd()}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple datasets named '{name}' found under {Path.cwd()}:\n"
            + '\n'.join(str(m) for m in matches)
        )
    return matches[0]


def run(dataset=SETTINGS.DATASET):
    # hdf5_file = Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/28_10_2500.h5')
    viewer = napari.Viewer()
    qt_window = viewer.window._qt_window  # access the underlying QMainWindow
    qt_window.resize(1200, 800)  # or whatever fits your VM screen

# Optionally, set minimum and maximum sizes so the user can still resize freely
    qt_window.setMinimumSize(800, 600)
    qt_window.setMaximumSize(1920, 1080)
    
    all_cells = AllCellsViewer(viewer, dataset)
    napari.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch the PhagoPred GUI.')
    parser.add_argument(
        'dataset',
        nargs='?',
        default=None,
        help='Name of the dataset .h5 file to search for in the current directory (defaults to SETTINGS.DATASET).',
    )
    args = parser.parse_args()

    if args.dataset is None:
        run()
    else:
        run(find_dataset(args.dataset))