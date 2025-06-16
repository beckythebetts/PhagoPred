import xarray
import numpy as np

from PhagoPred.tracking import tracking

def frame_to_frame_matching_test(num_cells=20):
    """
    Test if xarrays corrresponding to cells at consecutive frames give correct matching
    """
    initial_values = np.random.randint(low=0, high=100, size=(num_cells, 2))
    initial_idxs = np.random.default_rng().choice(
        np.arange(2*num_cells),
        (num_cells, ),
        replace=False
    )

    idx_swap = np.random.default_rng().choice(
        np.arange(num_cells),
        (num_cells, ),
        replace=False
    )

    idxs = [initial_idxs, initial_idxs[idx_swap]]

    frames = [xarray.DataArray(
        data=np.concatenate((idxs[i][:, np.newaxis], initial_values+i), axis=1),
        dims=('Cell Index', 'Feature'), 
        coords={"Feature": ["idx", "x", "y"]})
        for i in range(2)]
    
    lut, reindexed_frame1 = tracking.Tracking(file=None).frame_to_frame_matching(frames[0], frames[1], min_dist=5)
    assert(np.array_equal(reindexed_frame1.sel(Feature='idx').values, frames[0].sel(Feature='idx').values))
    print(reindexed_frame1, frames[0])

def main():
    frame_to_frame_matching_test()

if __name__ == '__main__':
    main()
    
    