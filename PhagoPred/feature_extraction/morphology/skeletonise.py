from skimage.morphology import skeletonize
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from PhagoPred import SETTINGS

def test(dataset=SETTINGS.DATASET, frame: int=0):
    with h5py.File(dataset, 'r') as f:
        test_mask = f['Segmentations']['Phase'][frame]
        cell_idxs = np.arange(0, np.max(test_mask) + 1)  # include max index
        
        skeletonised_im = np.zeros_like(test_mask, dtype=bool)

        expanded_mask = test_mask[np.newaxis, :, :] == cell_idxs[:, np.newaxis, np.newaxis]
        for mask in tqdm(expanded_mask):
            skeleton = skeletonize(mask)
            skeletonised_im = np.logical_or(skeletonised_im, skeleton)
    
    test_mask = np.where(test_mask>=0, 1, 0)
    # Plot overlay
    plt.figure(figsize=(8,8))
    plt.imshow(test_mask, cmap='gray')
    # Overlay skeleton in red
    plt.imshow(skeletonised_im, cmap='Reds', alpha=0.7)
    plt.axis('off')

    # Save overlay image
    save_path = Path('temp') / 'skeleton_overlay.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

    print(f"Saved skeleton overlay image to {save_path}")
    
if __name__ == '__main__':
    test()