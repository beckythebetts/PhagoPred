from pathlib import Path
import h5py

# ******* GENERAL *******
DATASET = Path('PhagoPred')/'Datasets'/'filter01_short (copy).h5'
MASK_RCNN_MODEL = Path("Models") / 'filter02'
CELLPOSE_MODEL = Path("PhagoPred") / 'cellpose_segmentation' / 'Models' / '5ims'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
REMOVE_EDGE_CELLS = True
with h5py.File(DATASET, 'r') as f:
    NUM_FRAMES = f['Images'].attrs['Number of frames']
    IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
    TIME_STEP = f['Images'].attrs['Time interval / s']
#IMAGE_SIZE = [2048, 2048]

# ******* EPI THRESHOLDING *******
THRESHOLD = 250

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 8
TRACK = True
CLEAN_TRACKS = True
MINIMUM_TRACK_LENGTH = 40
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 50 # Set as None to view all (slow)

# ******* FEATURE EXTRACTION *******
BATCH_SIZE = 50
PLOT_FEATURES = False
TRACKS_PLOT = True
SHOW_EATING = True
NUM_FRAMES_EATEN_THRESHOLD = 10
MINIMUM_PIXELS_PER_PATHOGEN = 10


# ******* MASK R-CNN MODEL TRAINING DIRECTORY STRUCTURE *******
# - 'Models'
#   - model name
#       - 'Training_Data'
#           - 'train'
#               - 'Images' .jpegs
#               - 'labels.json'
#           - 'validate'
#               - 'Images' .jpegs
#               - 'labels.json'

# ******* CELLPOSE MODEL TRAINING DIRECTORY STRUCTURE *******
# - 'cellpose_Models'
#   - model name
#           - 'train'
#               - *im.png
#               - *mask.png
#           - 'validate'
#               - *im.png
#               - *mask.png