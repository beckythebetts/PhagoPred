from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.extract_features import CellType

def flag_bad_frames(dataset: Path = SETTINGS.DATASET, use_features: list = ['Area', 'Perimeter'], threshold_factor: float = 2) -> None:
    """For each cell trajectory, get frames where use_features values spike or dip sharply.
    
    Args
    ----
        threshold_factor: residuals which are greater than threshold_factor*std(residuals) are flagged.
        
    """
    with h5py.File(dataset, 'r') as f:
        features_ds = CellType('Phase').get_features_xr(f, use_features)
        bad_frames = xr.zeros_like(features_ds[use_features[0]], dtype=bool)
        for feature_name in tqdm(use_features, desc='Flagging frames for removal'):
            ds = features_ds[feature_name]
            smoothed_ds = ds.rolling(Frame=5, center=True, min_periods=3).mean()
            residuals = np.abs(ds - smoothed_ds)
            
            stds = residuals.std(dim='Frame', skipna=True)
            thresholds = threshold_factor*stds
            
            thresholds = thresholds.expand_dims({'Frame': ds.sizes['Frame']}).transpose(*ds.dims)
            
            feature_bad_frames = residuals > thresholds
            
            bad_frames = bad_frames | feature_bad_frames
    
        return bad_frames.compute()

def remove_bad_frames(dataset: Path = SETTINGS.DATASET, use_features: list = ['Area', 'Perimeter'], threshold_factor: float = 2) -> None:
    bad_frames = flag_bad_frames(dataset, use_features, threshold_factor)
    with h5py.File(dataset, 'r+') as f:
        features_ds = f['Cells']['Phase']
        for feature_name in tqdm(features_ds.keys(), desc='Removing flagged frames from Cells dataset'):
            if features_ds[feature_name].shape[0] > 1:
                features_ds[feature_name][:] = np.where(bad_frames.values, np.nan, features_ds[feature_name][:])
        
        segmentations_ds = f['Segmentations']['Phase']
        for frame in tqdm(range(segmentations_ds.shape[0]), desc='Removing flagged frames from Segmentaion dataset'):
            bad_cells = np.nonzero(bad_frames.sel(Frame=frame).values)[0]
            frame_seg = segmentations_ds[frame][:]
            mask = np.isin(frame_seg, bad_cells)
            frame_seg[mask] = -1
            segmentations_ds[frame] = frame_seg
        
        features_ds.attrs['Cleaning Threshold'] = threshold_factor
            
def main():
    remove_bad_frames()
    
if __name__ == '__main__':
    main()
                
        

            
            
                             
        
        
            