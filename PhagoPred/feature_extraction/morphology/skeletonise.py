from skimage.morphology import skeletonize, binary_dilation
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from PhagoPred import SETTINGS


def test(dataset, frame: int = 0):
    with h5py.File(dataset, 'r') as f:
        test_mask = f['Segmentations']['Phase'][frame]
        cell_idxs = np.arange(0, np.max(test_mask) + 1)

        skeletonised_im = np.zeros_like(test_mask, dtype=bool)

        expanded_mask = test_mask[np.newaxis, :, :] == cell_idxs[:, np.newaxis,
                                                                 np.newaxis]
        for mask in tqdm(expanded_mask):
            skeleton = skeletonize(mask)

            skeletonised_im = np.logical_or(skeletonised_im, skeleton)
    skeletonised_im = binary_dilation(skeletonised_im)
    test_mask = np.where(test_mask >= 0, 255, 0)
    # test_mask = np.stack([test_mask] * 3, axis=0)
    # test_mask[skeletonised_im] = [255, 0, 0]
    # Plot overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(test_mask, cmap='gray')
    # Overlay skeleton in red
    plt.imshow(skeletonised_im, cmap='Reds', alpha=0.7)
    plt.axis('off')

    # Save overlay image
    save_path = Path('temp') / 'skeleton_overlay.jpg'
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    # plt.imsave(save_path, test_mask, dpi=200)
    plt.show()

    print(f"Saved skeleton overlay image to {save_path}")


if __name__ == '__main__':
    test(
        dataset=
        'C:\\Users\\php23rjb\\Documents\\PhagoPred\\PhagoPred\\Datasets\\06_03\\H.h5',
        frame=75)
