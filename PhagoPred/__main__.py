from pathlib import Path
import shutil
import os
import sys

from PhagoPred.utils.dataset_creation import process_ome_tiffs, find_ome_positions, preprocessing
from PhagoPred.utils.registration import register_frames
from PhagoPred.cellpose_segmentation.segment import seg_dataset
from PhagoPred.tracking.trackpy_2_stage import run_tracking
from PhagoPred.feature_extraction.extract_features import extract_features
from PhagoPred.death_classifier.pipeline import predict_bad_tracks, predict_death
from PhagoPred.utils.h5_file import H5File

DATA_DIR = Path('~/thor_server/MacrophageData').expanduser()

RUN_ID = 0


def process(dataset_dir: Path, tiffs_dir_name: str = '_1') -> None:
    local_dataset_dir = Path('PhagoPred') / 'Datasets' / dataset_dir.name
    local_dataset_dir.mkdir(parents=True, exist_ok=True)

    tiffs_dir = dataset_dir / tiffs_dir_name
    for pos in find_ome_positions(tiffs_dir, ['4', '5', '6']):
        print(f'Processing position {pos}')

        final_file = dataset_dir / f'{pos}.h5'
        if H5File(final_file).get_attr('Run ID') == RUN_ID:
            continue

        local_file = local_dataset_dir / final_file.name

        shutil.copy(final_file, local_file)
        # process_ome_tiffs(tiffs_dir, local_dataset_dir, positions=pos)
        # preprocessing(local_file)
        register_frames(local_file)
        seg_dataset(local_file)
        run_tracking(local_file)
        extract_features(local_file)
        predict_death(local_file)
        predict_bad_tracks(local_file)
        H5File(local_file).write_attr('Run ID', RUN_ID)
        H5File(local_file).repack()

        _copy_file_safe(local_file, final_file)


def _copy_file_safe(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(src.suffix + '.tmp')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    src.unlink()


def main():
    dirs = sys.argv[1:]
    for dir in dirs:
        process(DATA_DIR / dir)


if __name__ == '__main__':
    main()
