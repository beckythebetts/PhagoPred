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

import mask_funcs
from PhagoPred import SETTINGS
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MissingCell:
    def __init__(self, mask):
        self.mask = mask
        self.missing_count = 0

class Tracker:
    def __init__(self, name):
        self.name = name
        self.file = h5py.File(SETTINGS.DATASET, 'r+')
        self.frames_list = list(self.file['Segmentations']['Phase'].keys())
        self.frames_list.sort()
        self.images_list = list(self.file['Segmentations']['Phase'].keys())
        self.images_list.sort()
        self.old_frame = self.read_frame(0)
        self.new_frame = self.read_frame(1)
        self.max_index = torch.max(self.old_frame)
        # self.mask_ims = sorted([mask for mask in (SETTINGS.DIRECTORY / 'segmented' / self.name).iterdir()])
        # self.images = sorted([image for image in (SETTINGS.DIRECTORY / 'inference_dataset' / self.name).iterdir()])
        # self.old_frame = torch.tensor(utils.read_tiff(self.mask_ims[0]).astype(np.int16)).cuda()
        # self.new_frame = torch.tensor(utils.read_tiff(self.mask_ims[1]).astype(np.int16)).cuda()
        # self.max_index = torch.max(self.old_frame)
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
        #print('new ', len(mask_funcs.split_mask(self.new_frame, use_torch=True)))
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

        #old_mask_dict = mask_funcs.split_mask(self.old_frame, use_torch=True, return_indices=True)
        for missing_mask, missing_index in mask_funcs.SplitMask(self.old_frame):
            if missing_index not in self.missing_cells.keys():
                self.missing_cells[missing_index] = MissingCell(missing_mask)
        self.new_frame = updated_new_frame
        #print('memory', torch.cuda.memory_allocated(0), len(self.missing_cells))
        self.memory_file.write(f'{torch.cuda.memory_allocated(0)}\t{len(self.missing_cells)} \n')

    def track(self):
        print('\n--------------------\nTRACKING - ', self.name, '\n--------------------')
        #utils.remake_dir(SETTINGS.DIRECTORY / 'tracked' / self.name)
        # im = Image.fromarray(self.old_frame.cpu().numpy().astype(np.int16))
        # im.save(SETTINGS.DIRECTORY / 'tracking' / self.name / ("{0:03}".format(0) + '.tif'))
        #utils.save_tiff(self.old_frame.to(dtype=torch.int16).cpu().numpy().astype(np.uint16),SETTINGS.DIRECTORY / 'tracked' / self.name / ("{0:04}".format(0) + '.tif'))

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
            #utils.save_tiff(self.old_frame.to(dtype=torch.int16).cpu().numpy().astype(np.uint16), SETTINGS.DIRECTORY / 'tracked' / self.name / ("{0:04}".format(i) + '.tif'))


    def clean_up(self, threshold=SETTINGS.MINIMUM_TRACK_LENGTH):
        print('\n----------\nCLEANING TRACKS\n----------\n')
        # Removinf cells which are seen for < threshold number of frames
        # self.tracked_masks = sorted([mask for mask in (SETTINGS.DIRECTORY / 'tracked' / self.name).iterdir()])
        # length_of_tracks = {index : 0 for index in range(1, self.max_index+1)}
        length_of_tracks = {}
        for i in range(len(self.frames_list)):
        #for i, frame_path in enumerate(self.tracked_masks):
            sys.stdout.write(
                f'\rReading frame {i + 1} / {len(self.frames_list)}')
            sys.stdout.flush()
            frame = self.read_frame(i)
            # frame = torch.tensor(utils.read_tiff(frame_path).astype(np.int16)).cuda()
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
        #print(index_mapping)
        for i in range(len(self.frames_list)):
        # for i, frame_path in enumerate(self.tracked_masks):
            # BATCHES NEEDED TO SPEED THIS BIT UP
            sys.stdout.write(
                f'\rCleaning frame {i + 1} / {len(self.frames_list)}')
            sys.stdout.flush()
            frame = self.read_frame(i)
            # frame = torch.tensor(utils.read_tiff(frame_path).astype(np.int16)).cuda()
            # cleaned_frame = frame.clone()
            cleaned_frame = torch.zeros(tuple(SETTINGS.IMAGE_SIZE))
            for old_i, new_i in index_mapping.items():
                cleaned_frame[frame==old_i] = new_i
                #cleaned_frame = torch.where(frame==old_i, new_i, cleaned_frame)
            # for track in tracks_to_remove:
            #     cleaned_frame[frame == track] = 0
            self.write_frame(i, cleaned_frame.cpu())
            #utils.save_tiff(cleaned_frame.to(dtype=torch.int16).cpu().numpy().astype(np.uint16), frame_path)

    def show_tracks(self, num_frames=None):
        print('\n--------------------\nSHOWING TRACKS - ', self.name, '\n--------------------')
        #self.tracked_masks = sorted([mask for mask in (SETTINGS.DIRECTORY / 'tracked' / self.name).iterdir()])
        view_track_dir = SETTINGS.DATASET.parent / (SETTINGS.DATASET.stem + 'view_tracks')
        utils.remake_dir(view_track_dir)
        #total_num_cells = np.max(utils.read_tiff(self.tracked_masks[-1]))
        #colours = torch.tensor(np.random.uniform(0, 1, size=(total_num_cells+1, 3))).cuda()
        colour_dict = {}
        if num_frames is None:
            num_frames = len(self.images_list)
        for i in range(num_frames):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {num_frames}')
            sys.stdout.flush()
            mask = self.read_frame(i)
            image = torch.tensor(self.file['Images']['Phase'][self.images_list[i]][()].astype(np.int16)).to(
                device)
            im_rgb = torch.stack((image, image, image), axis=0)
            for j in range(torch.max(mask)):
                if j+1 in mask:
                    if j+1 not in colour_dict.keys():
                        colour_dict[j+1] = torch.tensor(np.random.uniform(0, 2**(8)-1, size=3)).cuda()
                    single_mask = torch.where(mask==j+1, 1, 0)
                    outline = mask_funcs.mask_outline(single_mask, 3)
                    for c in range(3):
                        im_rgb[c] = torch.where(outline, colour_dict[j+1][c], im_rgb[c])
            im_rgb = im_rgb.permute(1, 2, 0)
            utils.save_tiff((im_rgb).cpu().numpy().astype(np.uint8), view_track_dir / ("{0:04}".format(i) + '.jpg'))
        self.file.close()


def main():
    my_tracker = Tracker('Phase')
    my_tracker.track()
    if SETTINGS.CLEAN_TRACKS:
        my_tracker.clean_up()
    if SETTINGS.VIEW_TRACKS:
        my_tracker.show_tracks(SETTINGS.NUM_FRAMES_TO_VIEW)
    my_tracker.close()


if __name__ == '__main__':
    main()