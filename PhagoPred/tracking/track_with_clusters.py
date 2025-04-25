import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from scipy.optimize import linear_sum_assignment
import torch

from PhagoPred.utils import tools
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# for macropahges and bacteria:
    # get_tracklets (individual cells only)
    # join tracklets

# if cell containing bacteria disapears within ~1 cell radius of a cluster and reappears within ~1 cell radius containing same bacteria: join cells
