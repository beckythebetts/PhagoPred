from typing import Union

import h5py

class DatasetModel:
    def __init__(self, filepath):
        self.filepath = filepath

    def open(self):
        self.h5 = h5py.File(self.filepath, 'r')
        self.phase_data = self.h5['Images']['Phase']
        self.mask_data = self.h5['Segmentations']['Phase']
        self.X_centers = self.h5['Cells']['Phase']['X']
        self.Y_centers = self.h5['Cells']['Phase']['Y']
        self.num_frames = self.phase_data.shape[0]
        self.num_cells = self.X_centers.shape[1]

    def close(self):
        self.h5.close()

    def get_frame_data(self, frame_idx):
        phase = self.phase_data[frame_idx]
        mask = self.mask_data[frame_idx]
        x_centers = self.X_centers[frame_idx]
        y_centers = self.Y_centers[frame_idx]
        return phase, mask, x_centers, y_centers
    

