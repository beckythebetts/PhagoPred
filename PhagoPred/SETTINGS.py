from pathlib import Path

# ******* SEGMENTATION MODELS *******
MASK_RCNN_MODEL = Path(
    "PhagoPred") / 'detectron_segmentation' / 'models' / 'bio_20x_thp1'
CELLPOSE_MODEL = Path(
    '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1_clahe_withrescale'
)

# ******* TRACKING *******
MAXIMUM_DISTANCE_THRESHOLD = 60
FRAME_MEMORY = 8
MINIMUM_TRACK_LENGTH = 50

# ******* UMAP FEATURE EXTRACTION *******
NUM_TRAINING_FRAMES = 50
NUM_CONTOUR_POINTS = 50
PCA_COMPONENTS = 10
KMEANS_CLUSTERS = 12

DATASET = None
NUM_FRAMES = None
IMAGE_SIZE = None
TIME_STEP = None

ALL_FEATURES = [
    'Area',
    'Circularity',
    'Displacement',
    'Perimeter',
    'Phagocytes within 100 pixels',
    'Phagocytes within 250 pixels',
    'Phagocytes within 500 pixels',
    'Skeleton Branch Length Mean',
    'Skeleton Branch Length Max',
    'Skeleton Branch Length Std',
    'Skeleton Branches',
    'Skeleton Length',
    'Speed',
    'Major Axis Length',
    'Minor Axis Length',
    'Eccentricity',
    'Fluor Asymmetry',
    'Fluor CV',
    'Fluor Dist Mean',
    'Fluor Dist Std',
    'Fluor Mean',
    'Fluor Total',
]
