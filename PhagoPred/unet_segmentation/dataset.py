import torch
from PIL import Image
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from PhagoPred.utils import tools

class UNetDataset_train(torch.utils.data.Dataset):
    def __init__(self, ims_dir, masks_dir):
        self.image_paths = [im for im in ims_dir.iterdir()]
        self.mask_paths = [mask for mask in masks_dir.iterdir()]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = torch.tensor(np.array(image)).unsqueeze(0).float() / 255
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = torch.tensor(np.array(mask)).unsqueeze(0).float() / 255
        return tools.standardise(tools.pad_image(image)[0]), tools.pad_image(mask)[0]
    
class UNetDataset_inference(torch.utils.data.IterableDataset):
    def __init__(self, hdf5_dataset):
        self.im_dataset = hdf5_dataset['Images']['Phase']

    def __iter__(self):
        for im in self.im_dataset.keys():
            padded_im, self.padding_width, self.padding_height = tools.pad_image(torch.tensor(self.im_dataset[im][:]).unsqueeze(0).float() / 255)
            yield tools.standardise(padded_im)

def view_sample_ims(ims_dir, masks_dir, hdf5):
    train_dat = UNetDataset_train(ims_dir, masks_dir)
    train_im = train_dat[0][0].numpy()[0,:,:]
    # train_hist, train_bins = exposure.histogram(train_im)

    with h5py.File(hdf5, 'r') as f:
        inf_dat = iter(UNetDataset_inference(f))
        inf_im = next(inf_dat).numpy()[0,:,:]
    
    joined_im  = np.concatenate((train_im, inf_im), axis=1)
    plt.matshow(joined_im)
    plt.show()

def view_histograms(ims_dir, masks_dir, hdf5):
    train_dat = UNetDataset_train(ims_dir, masks_dir)
    train_im = train_dat[0][0].numpy()[0,:,:]
    # train_hist, train_bins = exposure.histogram(train_im)

    with h5py.File(hdf5, 'r') as f:
        inf_dat = iter(UNetDataset_inference(f))
        inf_im = next(inf_dat).numpy()[0,:,:]
    # inf_hist, inf_bins = exposure.histogram(inf_im)
    print(np.min(train_im), np.max(train_im), np.min(inf_im), np.max(inf_im))

    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(2, 4, tight_layout=True)
    axs[0, 0].hist(train_im.ravel(), bins=256)
    axs[0, 2].hist(inf_im.ravel(), bins=256)
    axs[1, 0].hist(tools.standardise(train_im).ravel(), bins=256)
    axs[1, 2].hist(tools.standardise(inf_im).ravel(), bins=256)

    # Plot matshow images and collect the matshow objects
    im1 = axs[0, 1].matshow(train_im, vmin=0, vmax=1)
    axs[0, 1].set_title('Training Image')
    im2 = axs[0, 3].matshow(inf_im, vmin=-0, vmax=1)
    axs[0, 3].set_title('Inference Image')
    im3 = axs[1, 1].matshow(tools.standardise(train_im), vmin=-5, vmax=5)
    axs[1, 1].set_title('Training Image - Standardised')
    im4 = axs[1, 3].matshow(tools.standardise(inf_im), vmin=-5, vmax=5)
    axs[1, 3].set_title('Inference Image - Standardised')

    # Create a colorbar shared across all matshow subplots
    # cbar = fig.colorbar(im1, ax=axs[:, 1:], orientation='vertical')
    # cbar.set_label('Intensity')

    plt.show()




def main():
    view_histograms(Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8' / 'Training_Data' / 'train' /'images',
                    Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8' / 'Training_Data' / 'train' /'masks',
                    Path('PhagoPred') / 'Datasets' / 'mac_short_im_only.h5')

if __name__ == '__main__':
    main()