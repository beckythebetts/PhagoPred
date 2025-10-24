import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy import interpolate
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs

class PreProcessing:

    def __init__(self, contour, mean_contour=None):
        self.contour=contour
        self.mean_contour = mean_contour
    
    # def smooth_contour(self, 
    def resample_contour(self, num_points=SETTINGS.NUM_CONTOUR_POINTS, smooth=0):
        """
        Set number of coordinates in contour to num_points
        """
        # Compute cumulative arc lengths
        distances = np.sqrt(np.sum(np.diff(self.contour, axis=0)**2, axis=1))
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)

        # Create equally spaced arc length values
        target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)

        if smooth > 0:
            tck, _ = interpolate.splprep([self.contour[:, 0], self.contour[:, 1]], s=smooth, per=True)
            x_resampled, y_resampled = interpolate.splev(np.linspace(0, 1, num_points), tck)
        else:
        # Interpolate contour points
            x_resampled = np.interp(target_lengths, cumulative_lengths, self.contour[:, 0])
            y_resampled = np.interp(target_lengths, cumulative_lengths, self.contour[:, 1])

        self.contour = np.column_stack((x_resampled, y_resampled))

    def centre_contour(self):
        """
        Adjust coordinates such that mean is at (0,0)
        """
        means = np.mean(self.contour, axis=0)
        self.contour = self.contour - means[np.newaxis, :]

    def normalise_contour_size(self):
        """
        Divide all coords by characteristic length scale.
        $R = \sqrt{\frac{\sum_{i=1}^{N}(x_{i}^{2}+{y}_{i}^{2})}{N}}$
        """
        R = np.sqrt(np.sum(self.contour[:, 0]**2 + self.contour[:, 1]**2) / SETTINGS.NUM_CONTOUR_POINTS)
        self.contour = self.contour / R
    
    def procrustes_align_contour(self):
        """Align contour with mean contour."""
        if self.mean_contour is not None:
            self.contour = np.reshape(self.contour, (-1, 2))
            _, self.contour, _ = procrustes(self.mean_contour, self.contour)
            self.contour = self.contour.flatten()

    def align_contour(self):
    
        """
        Approxiximate long axis by finding pair of coords with maximum distance between them. Rotate contour such that long axis is along x-axis.
        Shape can be mirrored in x or y axis, and will give different representation.
        So: Ensure largest |x| value is in +ve x region, and largest |y| is in +ve y region.

        Start contour at y=0, x>0, and move clockwise (all contours intitally have consistsent direction due to skimages contour algorithm).
        """
        distances = np.linalg.norm(self.contour[:, np.newaxis] - self.contour[np.newaxis], axis=2)
        max_idxs = np.unravel_index(np.argmax(distances), distances.shape)
        max_coord0, max_coord1 = self.contour[max_idxs[0]], self.contour[max_idxs[1]]
        theta = -np.arctan2(max_coord1[1] - max_coord0[1], max_coord1[0]-max_coord0[0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.contour = (R@self.contour.T).T

        #Reflect contour in x and or y axis, order of contour must be reversed if contour is flipped
        if np.max(-self.contour[:, 0]) > np.max(self.contour[:, 0]):
            self.contour[:, 0] = -self.contour[:, 0]
            self.contour = self.contour[::-1]

        if np.max(-self.contour[:, 1]) > np.max(self.contour[:, 1]):
            self.contour[:, 1] = -self.contour[:, 1]
            self.contour = self.contour[::-1]

        # Set start of contour to y ~=0, x>0
        potential_start_coords = self.contour.copy()
        potential_start_coords[potential_start_coords[:, 0]<0, 1] = np.nan
        first_coord = np.nanargmin(np.abs(potential_start_coords[:, 1]))
        self.contour = np.append(self.contour[first_coord:], self.contour[:first_coord], axis=0)

    def pre_process(self, smoothing=0):
        if len(self.contour) > 0:
            self.resample_contour(smooth=smoothing)
            self.centre_contour()
            self.normalise_contour_size()
            self.align_contour()
            self.procrustes_align_contour()
            return self.contour
        else:
            return []



def generalised_procrustes_analysis(contours, max_iter=20, tol=1e-7):
    """Align a list of contours iterativel yto mean shape
    """
    contours = [np.reshape(contour, (-1, 2)) for contour in contours]
    mean_shape = contours[0]

    for iteration in tqdm(range(max_iter), desc='Aligning contours'):
        aligned_contours = []

        # Align all contours to the current mean shape
        for contour in contours:
            _, aligned, _ = procrustes(mean_shape, contour)
            aligned_contours.append(aligned)

        # Compute new mean shape by averaging aligned contours
        new_mean_shape = np.mean(np.array(aligned_contours), axis=0)

        # Normalize mean shape (optional but common)
        new_mean_shape -= np.mean(new_mean_shape, axis=0)  # center
        scale = np.sqrt(np.sum(new_mean_shape**2) / new_mean_shape.shape[0])
        new_mean_shape /= scale

        # Check convergence
        diff = np.linalg.norm(mean_shape - new_mean_shape)
        print(f'GPA Iteration {iteration+1}, Mean shape change: {diff:.6f}')
        # if diff < tol:
        #     break

        mean_shape = new_mean_shape
    
    # plt.plot(mean_shape[:,0], mean_shape[:,1])
    # plt.show()
        
    aligned_contours = [aligned_contour.flatten() for aligned_contour in aligned_contours]
    
    return aligned_contours, mean_shape

def test():
    frame = 0
    cell_idx = 0
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        im = f['Segmentations']['Phase'][frame][:]
        # cell_mask = im[im==cell_idx]
        cell_mask = np.where(im==cell_idx, 1, 0)
        plt.imshow(cell_mask)
        # num_cells = f['Cells']['Phase'][:].shape[1]
        # contours = mask_funcs.get_border_representation(im, num_cells, f, 0)
        contours = mask_funcs.get_border_representation(cell_mask[np.newaxis, :, :])
        
        contour0 = contours[0]
        contour0 = np.array(contour0)
        print(contour0.shape)
        contour0[:, 0] = -contour0[:, 0]
        contour0 = contour0[::-1]
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(contour0[:, 0], contour0[:, 1])
        axs[0].set_aspect('equal')
        Pre = PreProcessing(contour0)
        Pre.pre_process()
        axs[1].plot(Pre.contour[:, 0], Pre.contour[:, 1])
        axs[1].axvline(x=0, color='k', linestyle='dashed')
        axs[1].axhline(y=0, color='k', linestyle='dashed')
        axs[1].plot(Pre.contour[:, 0][0], Pre.contour[:, 1][0], marker='$0$')
        axs[1].plot(Pre.contour[:, 0][1], Pre.contour[:, 1][1], marker='$1$')
        axs[1].set_aspect('equal')
        # plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    test()