import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
import sys
import torch
import gc
import cv2
import torch.nn.functional as F
import h5py

from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MissingCell:
    def __init__(self, mask):
        self.mask = mask
        self.missing_count = 0

class Tracker:
    """
    Re-indexes segmented phase channel such that, between two consecutive frame, each cell in frame, t, has the same index as the cell with which it has the greatest overlapin frame t+1, providied the overlap is greater than SETTINGS.OVERLAP_THRESHOLD.
    A cell with no found overlap is stored for SETTINGS.FRAME_MEMORY frames, and checked for matching cells.
    """
    def __init__(self):
        self.file = h5py.File(SETTINGS.DATASET, 'r+')
        self.frames_list = list(self.file['Segmentations']['Phase'].keys())
        self.frames_list.sort()
        self.images_list = list(self.file['Segmentations']['Phase'].keys())
        self.images_list.sort()
        self.old_frame = self.read_frame(0)
        self.new_frame = self.read_frame(1)
        self.max_index = torch.max(self.old_frame)
        self.missing_cells = {} # key is cell index, value is MissingCell class
        self.file['Segmentations']['Phase'].attrs['Overlap threshold'] = SETTINGS.OVERLAP_THRESHOLD
        self.file['Segmentations']['Phase'].attrs['Frame memory'] = SETTINGS.FRAME_MEMORY
        self.file['Segmentations']['Phase'].attrs['Minimum track length'] = SETTINGS.MINIMUM_TRACK_LENGTH
        self.memory_file = open('memory.txt', 'a')

    def read_frame(self, frame_index):
        return torch.tensor(self.file['Segmentations']['Phase'][self.frames_list[frame_index]][()].astype(np.int16)).to(device)

    def write_frame(self, frame_index, dataset):
        self.file['Segmentations']['Phase'][self.frames_list[frame_index]][...] = dataset

    def close(self):
        self.file.close()

    def add_missing_masks(self):
        for missing_index in self.missing_cells.keys():
            if missing_index not in self.old_frame:
                self.old_frame = torch.where(self.missing_cells[missing_index].mask.bool(), missing_index, self.old_frame)
            #self.old_frame += self.missing_cells[missing_index].mask*missing_index

    def update_new_frame(self):
        updated_new_frame = torch.zeros(tuple(SETTINGS.IMAGE_SIZE)).cuda()
        self.add_missing_masks()
        for new_mask, mask_index in mask_funcs.SplitMask(self.new_frame):
            # mask to check against = old_mask + missing_cell_masks
            intersection = torch.logical_and(new_mask, self.old_frame != 0)
            indexes, counts = torch.unique(self.old_frame[intersection], return_counts=True)
            if len(indexes) > 0 and torch.max(counts) > SETTINGS.OVERLAP_THRESHOLD*torch.sum(new_mask):
                new_index = indexes[torch.argmax(counts)]
                self.old_frame = torch.where(self.old_frame==indexes[torch.argmax(counts)], 0, self.old_frame)
                if new_index in self.missing_cells:
                    del self.missing_cells[new_index]
            else:
                new_index = self.max_index + 1
                self.max_index = new_index
            updated_new_frame += new_mask*int(new_index)

        for missing_index in list(self.missing_cells.keys()):
            self.missing_cells[missing_index].missing_count += 1
            if self.missing_cells[missing_index].missing_count >= SETTINGS.FRAME_MEMORY:
                if missing_index in self.old_frame:
                    self.old_frame = torch.where(self.old_frame==missing_index, 0, self.old_frame)
                del self.missing_cells[missing_index]
        for missing_mask, missing_index in mask_funcs.SplitMask(self.old_frame):
            if missing_index not in self.missing_cells.keys():
                self.missing_cells[missing_index] = MissingCell(missing_mask)
        self.new_frame = updated_new_frame
        self.memory_file.write(f'{torch.cuda.memory_allocated(0)}\t{len(self.missing_cells)} \n')

    def track(self):
        print('\n--------------------\nTRACKING\n--------------------')
        for i in range(1, len(self.frames_list)):
            sys.stdout.write(
                f'\rAdding frame {i+1} / {len(self.frames_list)}')
            sys.stdout.flush()

            self.new_frame = self.read_frame(i)
            #self.new_frame = torch.tensor(utils.read_tiff(self.mask_ims[i]).astype(np.int16)).cuda()
            self.update_new_frame()
            self.old_frame = self.new_frame
            self.write_frame(i, self.old_frame.cpu())
            gc.collect()


    def clean_up(self, threshold=SETTINGS.MINIMUM_TRACK_LENGTH):
        print('\n----------\nCLEANING TRACKS\n----------\n')
        # Removinf cells which are seen for < threshold number of frames
        length_of_tracks = {}
        for i in range(len(self.frames_list)):
            sys.stdout.write(
                f'\rReading frame {i + 1} / {len(self.frames_list)}')
            sys.stdout.flush()
            frame = self.read_frame(i)
            for index in torch.unique(frame):
                index = index.item()
                if index != 0:
                    if index not in length_of_tracks.keys():
                        length_of_tracks[index] = 0
                    length_of_tracks[index] += 1
        tracks_to_remove = torch.tensor(
            [index for index, track_length in length_of_tracks.items() if track_length < threshold]).cuda()
        index_mapping = {}
        new_index = 1
        for old_index in length_of_tracks.keys():
            if old_index not in tracks_to_remove:
                index_mapping[old_index] = new_index
                new_index += 1
        for i in range(len(self.frames_list)):
            sys.stdout.write(
                f'\rCleaning frame {i + 1} / {len(self.frames_list)}')
            sys.stdout.flush()
            frame = self.read_frame(i)
            cleaned_frame = torch.zeros(tuple(SETTINGS.IMAGE_SIZE))
            for old_i, new_i in index_mapping.items():
                cleaned_frame[frame==old_i] = new_i

            self.write_frame(i, cleaned_frame.cpu())


def main():
    my_tracker = Tracker()
    my_tracker.track()
    if SETTINGS.CLEAN_TRACKS:
        my_tracker.clean_up()
    my_tracker.close()


if __name__ == '__main__':
    main()