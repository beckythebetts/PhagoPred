import pyqtgraph as pg
from qtpy.QtCore import QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QPushButton, QSizePolicy,
    QScrollArea
)
import numpy as np
import h5py


class FeaturePlotsWidget(QWidget):
    """Widget to display feature plots with scrollable checkboxes for visibility."""
    def __init__(self, feature_names: list[str], parent_viewer=None):
        super().__init__()
        self.feature_names = feature_names
        self.parent_viewer = parent_viewer
        self.plot_widgets: dict[str, pg.PlotWidget] = {}

        # === MAIN LAYOUT ===
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # === TOGGLE BUTTON (top bar) ===
        self.toggle_button = QPushButton("⚙️ Show Feature Controls")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle_checkbox_panel)
        layout.addWidget(self.toggle_button)

        # === SCROLLABLE CHECKBOX PANEL ===
        self.checkbox_scroll = QScrollArea()
        self.checkbox_scroll.setWidgetResizable(True)
        self.checkbox_scroll.setVisible(False)
        self.checkbox_scroll.setMaximumHeight(0)  # Start collapsed
        layout.addWidget(self.checkbox_scroll)

        # Inner widget that holds all checkboxes
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setContentsMargins(10, 0, 10, 0)
        self.checkbox_scroll.setWidget(self.checkbox_widget)

        # Add checkboxes
        self.checkboxes: dict[str, QCheckBox] = {}
        for feat in feature_names:
            cb = QCheckBox(feat)
            cb.setChecked(True)
            cb.stateChanged.connect(self.toggle_feature_visibility)
            self.checkbox_layout.addWidget(cb)
            self.checkboxes[feat] = cb

        # === PLOT CONTAINER ===
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        layout.addWidget(self.plot_scroll)
        
        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container.setLayout(plot_layout)
        # layout.addWidget(self.plot_container)

        self.plot_scroll.setWidget(self.plot_container)
        first_pw = None
        for i, feat in enumerate(feature_names):
            pw = pg.PlotWidget()
            pw.setBackground('w')
            pw.showGrid(x=True, y=True)
            pw.setLabel('left', feat)
            if i != len(feature_names) - 1:
                pw.getAxis('bottom').setTicks([])  # Hide x-axis ticks
                pw.getAxis('bottom').setStyle(showValues=False)
            else:
                pw.setLabel('bottom', 'Frame')
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            plot_layout.addWidget(pw)
            self.plot_widgets[feat] = pw

            if i == 0:
                first_pw = pw
            else:
                pw.setXLink(first_pw)
        # === Animation for show/hide ===
        self.anim = QPropertyAnimation(self.checkbox_scroll, b"maximumHeight")
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)

    # --- Toggle visibility of the checkbox scroll panel ---
    def toggle_checkbox_panel(self):
        if self.checkbox_scroll.isVisible() and self.checkbox_scroll.maximumHeight() > 0:
            self.anim.setStartValue(self.checkbox_scroll.height())
            self.anim.setEndValue(0)
            self.anim.start()
            self.checkbox_scroll.setVisible(False)
            
            self.toggle_button.setText("⚙️ Show Feature To Plot")
        else:
            self.checkbox_scroll.setVisible(True)
            self.checkbox_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.anim.setStartValue(0)
            self.anim.setEndValue(600)  
            self.anim.start()
            self.toggle_button.setText("⚙️ Hide Feature To Plot")

    # --- Show/hide plots based on checkboxes ---
    def toggle_feature_visibility(self):
        for feat, cb in self.checkboxes.items():
            self.plot_widgets[feat].setVisible(cb.isChecked())

    # --- Update plots with new data ---
    def update_plots(self, f: h5py.File, cell_idx: int, first_frame: int, last_frame: int):
        x_data = np.arange(first_frame, last_frame)
        for feat, pw in self.plot_widgets.items():
            y_data = f['Cells']['Phase'][feat][first_frame:last_frame, cell_idx]
            pw.getPlotItem().clear()
            pw.plot(x_data, y_data, pen=pg.mkPen('k'))
