import sys

from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import ndimage
import torch

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_to(input: torch.tensor, max_h: int, max_w: int) -> torch.tensor:
    """Pad the given input to the target shape"""
    h, w = input.shape
    
    pad_h = max_h - h
    pad_w = max_w - w

    padded = torch.nn.functional.pad(input, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return padded

class Gabor:
    def __init__(self, wavelengths=np.arange(2, 15, 1), thetas=np.arange(0, 1, 0.25), gpu: bool=False):
        self.wavelengths = wavelengths
        self.frequencies = 1/self.wavelengths
        self.thetas = thetas*np.pi
        
        self.kernels = [
            gabor_kernel(frequency, theta) 
            for frequency in self.frequencies
            for theta in self.thetas
        ]
        self.kernel_pixels = [kernel.size for kernel in self.kernels] #num pixels per kernel for later normalising
        
        if gpu:
            self.kernels = [torch.tensor(kernel, dtype=torch.complex64, device=device) for kernel in self.kernels]
            
            max_h, max_w = max(kernel.shape[0] for kernel in self.kernels), max(kernel.shape[1] for kernel in self.kernels)
            self.padding = max(max_h, max_w)
            self.kernels = [pad_to(kernel, max_h, max_w) for kernel in self.kernels]
            
            self.kernels = torch.stack(self.kernels)[:, None, :, :]
            
            self.kernel_pixels = torch.tensor(self.kernel_pixels, device=device)
        
    
    def get_dominant_scales_batch(self, im_batch: torch.tensor, mask_batch: torch.tensor) -> torch.tensor:
        im_batch = im_batch.to(torch.complex64)
        im_batch = im_batch[:, None, :, :]
        filtered_ims = torch.nn.functional.conv2d(im_batch, self.kernels, padding=self.padding)
        filtered_ims[mask_batch[:, None, :, :]] = 0
        
        energies = torch.sum(torch.abs(filtered_ims**2)/self.kernel_pixels[None, :, None, None], dim=(2, 3))
        energies = energies.view(-1, len(self.frequencies), len(self.thetas))
        energies = torch.sum(energies, dim=2)
        
        relative_energies = energies / torch.sum(energies, dim=1)
        
        print(relative_energies)


    def view_kernels(self):
        fig, axs = plt.subplots(len(self.frequencies), len(self.thetas), figsize=(len(self.thetas),len(self.frequencies)))
        for i, wave_length in enumerate(self.wavelengths):
            for j, theta in enumerate(self.thetas):
                axs[i, j].matshow(np.real(self.kernels[i*len(self.thetas)+j]))

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                if j == 0:
                    axs[i, j].set_ylabel(f'{wave_length} pix')
                if i == 0:
                    axs[i, j].set_title(f'{theta/np.pi:.2f} pi rad')
        plt.tight_layout()
        plt.savefig('temp/example.png')

    def band_pass_filter(self, image_array, low=1, high=5):
        low_blur = ndimage.gaussian_filter(image_array, low)
        high_blur = ndimage.gaussian_filter(image_array, high)

        return low_blur - high_blur
    
    def apply_to_im(self, image_array):
        filtered = [ndimage.convolve(image_array, kernel) for kernel in self.kernels]
        return np.array(filtered)
    
    def get_relative_energies(self, image_array, mask):
        """
        Get energy (sum of abs(pixel values)**2) at each wavelength, and normalise
        """
        # image_array = self.band_pass_filter(image_array)
        filtered_images = self.apply_to_im(image_array)
        filtered_images[:, ~mask] = 0

        filter_num_pixels = [filter.size for filter in self.kernels]
        energies = np.array([np.sum(np.abs(filtered_image**2))/num_pixels for filtered_image, num_pixels in zip(filtered_images, filter_num_pixels)])
        energies = energies.reshape(len(self.frequencies), len(self.thetas))

        rotation_invariant_energies = np.sum(energies, axis=1)
        relative_energies = rotation_invariant_energies / np.sum(rotation_invariant_energies)

        return relative_energies
    
    def get_dominant_scales(self, image_array, masks):
        dominant_scales = np.full(len(masks), np.nan)
        if torch.is_tensor(image_array):
             image_array = image_array.cpu().numpy()
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        for i, mask in enumerate(masks):
            if mask.sum() > 10:
                mask = mask_funcs.erode_mask(mask)
                row_slice, column_slice = mask_funcs.get_minimum_mask_crop(mask)

                cropped_image_array = image_array[row_slice, column_slice].astype(float)
                mask = mask[row_slice, column_slice]

                relative_energies = self.get_relative_energies(cropped_image_array, mask)

                dominant_scales[i] = self.wavelengths[np.argmax(relative_energies)]

        return dominant_scales

    def view_convolved_im(self, image_array=0):
        if image_array == 0:
            image_array, mask = get_random_cell()
        
    
        fig, axs = plt.subplots(len(self.frequencies), len(self.thetas)+1, figsize=(len(self.thetas),len(self.frequencies)))
        filtered = self.apply_to_im(image_array)
        relative_energies = self.get_relative_energies(image_array, mask)
        for i, wave_length in enumerate(self.wavelengths):
            summed_filters = np.sum(np.real(filtered[i*len(self.thetas):(i+1)*len(self.thetas)]), axis=0)
            summed_filters[~mask] = np.nan
            axs[i, -1].matshow(summed_filters)
            axs[i, -1].set_xticks([])
            axs[i, -1].set_yticks([])
            for j, theta in enumerate(self.thetas):
                sys.stdout.write(f'\nConvolving image with filter {i*len(self.thetas)+j+1} of {len(self.frequencies*len(self.thetas))}')
                sys.stdout.flush()
                filtered_image = np.real(filtered[i*len(self.thetas)+j])
                filtered_image[~mask] = np.nan
                axs[i, j].matshow(filtered_image)

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                if j == 0:
                    axs[i, j].set_ylabel(f'{wave_length} pix')
                if i == 0 and j<=len(self.thetas):
                    axs[i, j].set_title(f'{theta/np.pi:.2f} pi rad')
        axs[0, -1].set_title('sum')
        plt.tight_layout()
        plt.savefig('temp/example.png')

    def view_relative_energies(self, num_ims=4):
        fig, axs = plt.subplots(3,  num_ims)
        for i in range(num_ims):
            image_array, mask = get_random_cell()

            masked_image = image_array.copy()
            masked_image[~mask] = np.nan
            
            axs[0, i].matshow(masked_image)
            axs[0, i].axis('off')

            # band_pass_filtered_image = self.band_pass_filter(image_array)
            axs[1, i].matshow(image_array)
            axs[1, i].axis('off')

            relative_energies = self.get_relative_energies(image_array, mask)
            axs[2, i].bar(self.wavelengths, relative_energies)
            axs[2, i].set_xlabel('wavelength / \npixels')
            axs[2, i].grid()
            # axs[1, i].set_ylim(0, 1)
            if i > 0:
                axs[2, i].sharey(axs[2, 0])
                # axs[1, i].set_yticklabels([])

            axs[0, i].set_title(f'Scale:\n{self.wavelengths[np.argmax(relative_energies)]}')
        axs[2, 0].set_ylabel('relative energy')
        plt.tight_layout()
        plt.savefig('temp/example.png')




def get_random_cell(erosion_val=10):
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        frame = np.random.randint(0, SETTINGS.NUM_FRAMES)
        image = f['Images']['Phase'][f'{frame:04}'][:]
        mask = f['Segmentations']['Phase'][f'{frame:04}'][:]

        cell_idx = np.random.choice(np.unique(mask[mask>0]))

        mask = mask==cell_idx

        struct = np.ones((2*erosion_val+1, 2*erosion_val+1))  
        mask = ndimage.binary_erosion(mask, structure=struct)


        row_slice, col_slice = mask_funcs.get_minimum_mask_crop(mask)
        image, mask = image[row_slice, col_slice], mask[row_slice, col_slice]
        image_array = image.astype(float)

        if np.sum(mask) < 50:
            image_array, mask = get_random_cell()

        return image_array, mask

if __name__ == '__main__':
    test = Gabor(gpu=True)
    test.get_dominant_scales_batch(torch.randn(10, 50, 50), torch.randn(10, 50, 50))
    # test.view_relative_energies()
    # test.view_convolved_im()