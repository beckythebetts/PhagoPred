from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout
import napari
from qtpy.QtCore import QThread
import sys

from PhagoPred.display.GUI import model, presenter, view
from PhagoPred import SETTINGS

class MainWindow(QMainWindow):
    def __init__(self, dataset_path, start_frame=0, end_frame=100, step=1):
        super().__init__()

        self.frames_loaded = 0
        self.start_frame = start_frame
        self.end_frame = end_frame
        # Central widget with vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create napari viewer widget
        self.viewer = napari.Viewer(show=False)
        layout.addWidget(self.viewer.window.qt_viewer)  # add napari QWidget to your layout

        # Setup MVC components
        self.model = model.DatasetModel(dataset_path)
        self.view = view.AllCellsView(self.viewer)

        self.loading_dialog = view.LoadingBarDialog(max_value=(self.end_frame - self.start_frame))
        self.loading_dialog.show()

        self.presenter = presenter.FrameLoaderWorker(self.model, start_frame, end_frame, step)
        self._connectSignals()
        self.presenter.start()

        self.setWindowTitle("Napari Frame Viewer")
        self.resize(800, 600)

    def _connectSignals(self):
        self.presenter.frameLoaded.connect(self.on_frame_loaded)
        self.presenter.finshedSignal.connect(self.loading_dialog.close)

    def closeEvent(self, event):
        # Make sure to stop the thread nicely on close
        if self.presenter.isRunning():
            self.presenter.stop()
            self.presenter.wait()
        event.accept()

    def on_frame_loaded(self, frame_idx, data):
        self.frames_loaded += 1
        self.loading_dialog.update_progress(self.frames_loaded)
        self.view.update_frame(frame_idx, data)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(SETTINGS.DATASET, start_frame=0, end_frame=100, step=1)
    window.show()

    sys.exit(app.exec_())
