import h5py
import numpy as np
from pathlib import Path


def create_synthetic_hdf5(
    filename: Path,
    num_cells: int,
    num_frames: int,
    censor_rate: float,
    seed: int,
):
    np.random.seed(seed)
    with h5py.File(filename, "w") as f:
        cells = f.create_group("Cells")
        phase = cells.create_group("Phase")

        start_frames = np.random.randint(0, num_frames//3, size=num_cells)
        lifetimes = np.random.randint(250, 1000, size=num_cells)

        end_frames = np.minimum(start_frames + lifetimes, num_frames-1)

        dies = np.random.rand(num_cells) > censor_rate
        death_frames = np.where(dies,
                                end_frames,
                                np.nan)
        death_frames = np.where(~np.isnan(death_frames) & (death_frames > end_frames),
                                np.nan,
                                death_frames)

        def create_feature(name):
            return phase.create_dataset(
                name,
                shape=(num_frames, num_cells),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=True,
            )
        
        # for feature in features:
        #     create_feature(feature)

        area_ds = create_feature("Area")
        perim_ds = create_feature("Perimeter")
        circ_ds = create_feature("Circularity")
        speed_ds = create_feature("Speed")
        dens_ds  = create_feature("DensityPhase")
        disp_ds  = create_feature("Displacement")
        # cd_ds    = create_feature("CellDeath")   # just like real dataset

        # for c in range(num_cells):
        #     if np.isnan(death_frames[c]):
        #         cd_ds[:, c] = np.nan
        #     else:
        #         cd_ds[int(death_frames[c]), c] = 1.0

        cd_ds = phase.create_dataset(
            'CellDeath', 
            shape=(1, num_cells), 
            dtype=np.float32
            )
        cd_ds[:] = death_frames
        for c in range(num_cells):
            s = start_frames[c]
            e = int(end_frames[c])

            T = e - s + 1
            t = np.linspace(0, 1, T)

            dies = not np.isnan(death_frames[c])

            trend = np.zeros(T)
            if dies:
                # Only dying cells show an increasing risk trend
                # risk = np.abs(np.random.randn()) * 0.8       # strong positive trend
                # trend = risk * t**2          
                # # nonlinear increase toward death
                death_frame = int(death_frames[c])
                base_onset = 250
                jitter = np.random.randint(-20, 20)
                onset_frame = max(s, death_frame - base_onset + jitter)
                risk = np.abs(np.random.randn()) * 0.8
                risk = 1.0
                for i, frame in enumerate(range(s, e+1)):
                    if frame >= onset_frame:
                        # from 0 → 1 between onset and death
                        t_rel = (frame - onset_frame) / (death_frame - onset_frame + 1e-6)
                        trend[i] = risk * (t_rel ** 2)
            else:
                # Censored cells show no trend (neutral baseline)
                trend = np.random.randn() * 0.1 * np.ones_like(t)

            area = 80 + 8 * trend + np.random.randn(T) * 3
            perim = 40 + 5 * trend + np.random.randn(T)
            circ = 0.5 + 0.1 * trend + np.random.randn(T) * 0.05
            speed = np.abs(np.random.randn(T) * (0.3 + 0.2 * trend))
            dens = np.random.randn(T) * 0.5 - 2 * trend
            disp = np.cumsum(speed)

            area_ds[s:e+1, c]  = area
            perim_ds[s:e+1, c] = perim
            circ_ds[s:e+1, c]  = circ
            speed_ds[s:e+1, c] = speed
            dens_ds[s:e+1, c]  = dens
            disp_ds[s:e+1, c]  = disp

    print(f"✓ Created synthetic dataset: {filename}")
    return filename


if __name__ == '__main__':
    create_synthetic_hdf5(
        filename = Path('PhagoPred') / 'Datasets' / 'val_synthetic.h5',
        num_cells=1000,
        num_frames=1000,
        censor_rate=0.1,
        seed=42
    )