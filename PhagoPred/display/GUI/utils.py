from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QApplication

class LoadingBarDialog(QDialog):
    def __init__(self, max_value, message="Loading..."):
        super().__init__()
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()

        self.label = QLabel(message)
        self.progress = QProgressBar()
        self.progress.setRange(0, max_value)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def show(self):
        super().show()
        QApplication.processEvents()

    def update_progress(self, value, message=None):
        self.progress.setValue(value)
        if message:
            self.label.setText(message)
        QApplication.processEvents()  # Important: refresh UI