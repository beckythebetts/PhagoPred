import pywt
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage.transform import resize

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs

def get_wavelet_stats(im, mask, n_levels=5):
    means = []
    vars = []
    energies = []
    correlation_energies = []
    ims = []
    for _ in range(n_levels):
        im, (H, V, D) = pywt.dwt2(im, 'haar')
        mask = resize(mask, output_shape=im.shape)
        try:
            mean = np.sum([np.sum(detail[mask])/np.sum(~mask) for detail in (H, V, D)])
            var = np.sum([np.sum(detail[mask] - mean)**2/np.sum(~mask)**2 for detail in (H, V, D)])
            energy = np.sum([np.sum(detail[mask]**2)/np.sum(mask)**2 for detail in (H, V, D)])
            correlation_energy = np.sum([np.sum(detail[mask]**2)/np.sum(im[mask])**2 for detail in (H, V, D)])
        except:
            correlation_energy, mean, var, energy = np.nan, np.nan, np.nan, np.nan
        means.append(mean)
        vars.append(var)
        energies.append(energy)
        correlation_energies.append(correlation_energy)

        view_im = im.copy()
        view_im[~mask] = np.nan
        ims.append(view_im)
    # print(means, vars, energies)
    return correlation_energies, means, vars, energies, ims

def view_examples_grid(h5py_file=SETTINGS.DATASET, rows=3, columns=3, erosion_val=10):
    with h5py.File(h5py_file, 'r') as f:
        # num_examples = rows*columns
        fig, axs = plt.subplots(rows, columns)

        for i in np.arange(rows):
            for j in np.arange(columns):
                frame = np.random.randint(0, SETTINGS.NUM_FRAMES)
                image = f['Images']['Phase'][f'{frame:04}'][:]
                mask = f['Segmentations']['Phase'][f'{frame:04}'][:]

                # cell_idx = np.random.randint(1, np.max(mask))
                cell_idx = np.random.choice(np.unique(mask[mask>0]))

                mask = mask==cell_idx

                struct = np.ones((2*erosion_val+1, 2*erosion_val+1))  
                mask = binary_erosion(mask, structure=struct)


                row_slice, col_slice = mask_funcs.get_minimum_mask_crop(mask)
                image, mask = image[row_slice, col_slice], mask[row_slice, col_slice]
                image = image.astype(float)
                # image = image * mask
                masked_image = image.copy()
                masked_image[~mask] = np.nan

                means, vars, energies, ims = get_wavelet_stats(image, mask)

                axs[i, j].matshow(masked_image)

                # axs[1, i].plot(range(len(energies)), energies)
                axs[i, j].axis('off')
                
                # for j in range(3):
                #     axs[j+2, i].matshow(ims[j])
                #     axs[j+2, i].axis('off')
                axs[i, j].set_title(f'{np.argmax(means)}, {np.argmax(vars)}, {np.argmax(energies)}')

    plt.tight_layout()
    plt.savefig('temp/example.png')

def view_examples(h5py_file=SETTINGS.DATASET, ims=3, erosion_val=10):
    with h5py.File(h5py_file, 'r') as f:
        # num_examples = rows*columns
        fig, axs = plt.subplots(5, ims)

        for i in np.arange(ims):
            frame = np.random.randint(0, SETTINGS.NUM_FRAMES)
            image = f['Images']['Phase'][f'{frame:04}'][:]
            mask = f['Segmentations']['Phase'][f'{frame:04}'][:]

            # cell_idx = np.random.randint(1, np.max(mask))
            cell_idx = np.random.choice(np.unique(mask[mask>0]))

            mask = mask==cell_idx

            struct = np.ones((2*erosion_val+1, 2*erosion_val+1))  
            mask = binary_erosion(mask, structure=struct)


            row_slice, col_slice = mask_funcs.get_minimum_mask_crop(mask)
            image, mask = image[row_slice, col_slice], mask[row_slice, col_slice]
            image = image.astype(float)
            # image = image * mask
            masked_image = image.copy()
            masked_image[~mask] = np.nan

            corr_en, means, vars, energies, w_tra = get_wavelet_stats(image, mask)

            axs[0, i].matshow(masked_image)

            # axs[1, i].plot(range(len(corr_en)), corr_en, label='corr en')
            # axs[1, i].plot(range(len(means)), means, label='means')
            # axs[1, i].plot(range(len(vars)), vars, label='vars')
            axs[1, i].plot(range(len(energies)), energies, label='energies')
            # axs[1, i].legend()

            axs[0, i].axis('off')
            
            for j in range(3):
                axs[j+2, i].matshow(w_tra[j])
                axs[j+2, i].axis('off')
            print(corr_en)
            axs[0, i].set_title(f'maximum energy: \n{np.nanmax(energies):.0f} \nat scale \n{np.nanargmax(energies)}')
    plt.tight_layout()
    plt.savefig('temp/example.png')


if __name__ == '__main__':
    # view_examples_grid()
    view_examples()