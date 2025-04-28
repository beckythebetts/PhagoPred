from pathlib import Path
import h5py

# ******* GENERAL *******
# DATASET = Path('PhagoPred')/'Datasets'/'secondwithlight - Copy.h5'
DATASET = Path('PhagoPred')/'Datasets'/'mac_07_03_short (copy).h5'
# DATASET = Path('PhagoPred')/'Datasets'/'mac_short_seg.h5'
MASK_RCNN_MODEL = Path("PhagoPred") / 'detectron_segmentation' / 'models' / 'mac_20x'
CELLPOSE_MODEL = Path("PhagoPred") / 'cellpose_segmentation' / 'Models' / 'ash'
UNET_MODEL = Path("PhagoPred") / 'unet_segmentation' / 'models' / '20x_flir_8'
TRAINING_DATA = Path("PhagoPred") / 'segmentation' / 'models' / '20x_flir_8'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
REMOVE_EDGE_CELLS = True
with h5py.File(DATASET, 'r') as f:
    NUM_FRAMES = f['Images'].attrs['Number of frames']
    IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
    TIME_STEP = f['Images'].attrs['Time interval / s']
# MODEL_IMAGE_SIZE = [5472, 3648]

# ******* EPI THRESHOLDING *******
THRESHOLD = 250

# ******* TRACKING *******
MINIMUM_DISTANCE_THRESHOLD = 20
FRAME_MEMORY = 10
CLEAN_TRACKS = True
MINIMUM_TRACK_LENGTH = 50


VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 50 # Set as None to view all (slow)

# ******* FEATURE EXTRACTION *******
NUM_TRAINING_FRAMES = 10
NUM_CONTOUR_POINTS = 50
PCA_COMPONENTS = 10
KMEANS_CLUSTERS = 4


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