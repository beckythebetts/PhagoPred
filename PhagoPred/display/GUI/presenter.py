import numpy as np
from qtpy.QtCore import QObject, QThread, Signal

class FrameLoaderWorker(QThread):
    frameLoaded = Signal(int, object)  # frame index, data dict
    finshedSignal = Signal()

    def __init__(self, model, start_frame, end_frame, step=1):
        super().__init__()
        self.model = model
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.batch_size = 10
        self._running = True

    def run(self):
        self.model.open()
        frames = list(range(self.start_frame, self.end_frame, self.step))
        for i in range(0, len(frames), self.batch_size):
            if not self._running:
                break
            batch_frames = frames[i:i+min(i+self.batch_size, len(frames))]
            phase_batch, mask_batch, x_centers_batch, y_centers_batch = self.model.get_frame_data(batch_frames)
            # Prepare data (convert grayscale to RGB, prepare points)

            for idx_in_batch, frame in enumerate(batch_frames):
                phase = phase_batch[idx_in_batch]
                mask = mask_batch[idx_in_batch]
                x_centers = x_centers_batch[idx_in_batch]
                y_centers = y_centers_batch[idx_in_batch]

                # Convert grayscale to RGB
                phase_rgb = np.stack([phase]*3, axis=-1).astype(np.uint8)

                points = []
                labels = []

                # Loop over cells in this frame
                for cell_idx in range(len(x_centers)):
                    x, y = x_centers[cell_idx], y_centers[cell_idx]
                    if not np.isnan(x) and not np.isnan(y):
                        # napari expects (frame, y, x)
                        points.append([frame, y, x])
                        labels.append(cell_idx)

                points = np.array(points) if points else np.empty((0, 3))
                labels = np.array(labels) if labels else np.empty((0,))

                self.frameLoaded.emit(frame, {
                    'phase_rgb': phase_rgb,
                    'mask': mask,
                    'points': points,
                    'labels': labels,
                })
        self.finshedSignal.emit()
        self.model.close()

    def stop(self):
        self._running = False
