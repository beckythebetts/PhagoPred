import numpy as np
import torch
import h5py
import sys
from pathlib import Path
import imageio

from PhagoPred import SETTINGS
from PhagoPred.utils import tools
from PhagoPred.utils import mask_funcs
from PhagoPred.tracking.nearest_neighbour_tracking import Track, NearestNeighbourTracking

class PhagocyticEvent:
    def __init__(self, frames, pathogen_indices):
        self.frames = np.array(list(frames))
        self.pathogen_indices = np.array(list(pathogen_indices))

    def add_frame(self, frame, index):
        self.frames = np.append(self.frames, frame)
        self.pathogen_indices = np.append(self.pathogen_indices, index)

    def save_event(self, cell):
        if len(self.frames) >= SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
            with h5py.File(SETTINGS.DATASET, 'r+') as f:
                dtype = np.dtype([('frame', 'f4'), ('pathogen_index', 'f4')])
                data = np.zeros(self.frames.size, dtype=dtype)
                data['frame'] = self.frames
                data['pathogen_index'] = self.pathogen_indices
                try:
                    f.create_dataset(f'Features/{cell}/{int(self.frames[0])}_{int(self.frames[-1])}', data=data)
                except ValueError:
                    f.create_dataset(f'Features/{cell}/{int(self.frames[0])}_{int(self.frames[-1])}_1', data=data)
                if len(self.frames) != len(np.unique(self.frames)):
                    # print('PROBLEM FOUND', self.frames, self.pathogen_indices)
                    print('PROBLEM FOUND', cell)


def track_phagocytosis_events(hdf5file):
    with h5py.File(hdf5file, 'r+') as f:
        f['Features'].attrs['Threshold number of frames for phagocytosis'] = SETTINGS.NUM_FRAMES_EATEN_THRESHOLD
        f['Features'].attrs['Threshold number of pixels for phagocytosis'] = SETTINGS.MINIMUM_PIXELS_PER_PATHOGEN
        for cell in f['Features']:
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            phago_events = f['Features'][cell]['PhagocyticFrames']
            frames = phago_events['frame'][:]
            pathogen_indices = phago_events['pathogen_index'][:]
            if len(frames) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                sequences = tools.split_list_into_sequences(frames, return_indices=True)
                for sequence in sequences:
                    #print(sequence)
                    if len(sequence) > SETTINGS.NUM_FRAMES_EATEN_THRESHOLD:
                        # if only one pathogen, no need to track
                        if len(frames[sequence]) == len(set(frames[sequence])):

                            event = PhagocyticEvent(frames[sequence], pathogen_indices[sequence])
                            event.save_event(cell)
                        else:
                            # get pathogen_centres from epi masks:
                            centres = [mask_funcs.get_centre(np.where(f['Segmentations']['Epi'][f'{int(frame):04}'][:]==index, 1, 0)) for frame, index in zip(frames[sequence], pathogen_indices[sequence])]
                            # NN tracker
                            tracker = NearestNeighbourTracking(frames[sequence], pathogen_indices[sequence], centres)
                            tracker.track()
                            for track in tracker.tracked:
                                event = PhagocyticEvent(track.track_dict.keys(), track.track_dict.values())
                                event.save_event(cell)


def del_events(dataset):
    with h5py.File(dataset, 'r+') as f:
        for cell in f['Features'].keys():
            for phago_event in f['Features'][cell]:
                #print(phago_event)
                if phago_event != 'MorphologicalFeatures' and phago_event != 'PhagocyticFrames':
                    del(f['Features'][cell][phago_event])


def show_phagocytic_events(dataset, save_directory, frame_size=150):
    tools.remake_dir(Path(save_directory))
    with h5py.File(dataset, 'r') as f:
        for cell in f['Features'].keys():
            sys.stdout.write(f'\r{cell}')
            sys.stdout.flush()
            for phago_event in f['Features'][cell]:
                #print(phago_event)
                if phago_event != 'MorphologicalFeatures' and phago_event != 'PhagocyticFrames':
                    # del(f['Features'][cell][phago_event])
                    (Path(save_directory) / cell / phago_event).mkdir(parents=True)
                    frames = f['Features'][cell][phago_event]['frame'][:]
                    for frame in np.arange(np.max((min(frames)-10, 0)), np.min((max(frames)+11, SETTINGS.NUM_FRAMES))):
                        # find centre of cell, if nan, find previous centre ...
                        xcentre = np.nan
                        framei = frame
                        while np.isnan(xcentre):
                            xcentre, ycentre = f['Features'][cell]['MorphologicalFeatures'][int(framei)]['xcentre'], f['Features'][cell]['MorphologicalFeatures'][int(framei)]['ycentre']
                            framei -= 1
                        xmin, xmax, ymin, ymax = int(xcentre-(frame_size/2)), int(xcentre+(frame_size/2)), int(ycentre-(frame_size/2)), int(ycentre+(frame_size/2))
                        phase_image = np.array(f['Images']['Phase'][f'{int(frame):04}'][xmin:xmax, ymin:ymax])
                        phase_mask = np.array(f['Segmentations']['Phase'][f'{int(frame):04}'][xmin:xmax, ymin:ymax])
                        outline = mask_funcs.mask_outline(torch.where(torch.tensor(phase_mask) == int(cell[-4:]), 1, 0), thickness=2).cpu().numpy()
                        epi_image = np.array(f['Images']['Epi'][f'{int(frame):04}'][xmin:xmax, ymin:ymax])
                        if frame in frames:
                            pathogen_index = f['Features'][cell][phago_event]['pathogen_index'][np.argwhere(frames==frame)]
                            # print(pathogen_index)
                            epi_mask = np.where(f['Segmentations']['Epi'][f'{int(frame):04}'][xmin:xmax, ymin:ymax]==pathogen_index, 1, 0)
                        else:
                            epi_mask = np.zeros(phase_image.shape)
                        im_rgb = np.stack((phase_image, phase_image, phase_image), axis=0)

                        im_rgb[0] = np.where(outline, 255, im_rgb[0])
                        im_rgb[1] = np.where(outline, 255, im_rgb[1])
                        im_rgb[2] = np.where(outline, 0, im_rgb[2])

                        im_rgb[0] = np.where(epi_mask, epi_image, im_rgb[0])

                        im_rgb = im_rgb.transpose(1, 2, 0)

                        imageio.imwrite(Path(save_directory) / cell / phago_event / '{0:04}.png'.format(frame), im_rgb.astype(np.uint8))

def main():
    hdf5file = SETTINGS.DATASET
    # del_events(hdf5file)
    # track_phagocytosis_events(hdf5file)
    if SETTINGS.SHOW_EATING:
        show_phagocytic_events(hdf5file, str(SETTINGS.DATASET.parent / str(SETTINGS.DATASET.stem + 'show_eating')))

if __name__ == '__main__':
    main()
