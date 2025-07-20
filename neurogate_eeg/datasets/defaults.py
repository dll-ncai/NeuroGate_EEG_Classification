from .sc_pipeline import get_scratch_pl
from .sc_pipeline import get_scratch_updated_pl

import numpy as np

# Enter paths to the datasets here
tuh_dataset = r'/path/to/TUH EEG Corpus'
nmt_dataset = r'/path/to/[M] nmt_scalp_eeg_dataset'
nmt_4k_dataset = r'/path/to/nmt_4k_split'

def get_def_ds(mins = 1):
    return (tuh_dataset, get_scratch_pl('TUH', mins), np.array([1, 1]), "results/tuh"), \
            (nmt_dataset, get_scratch_updated_pl('NMT', mins), np.array([1900, 305]), "results/nmt"), \
            (nmt_4k_dataset, get_scratch_updated_pl('NMT', mins), np.array([3945, 708]), "results/nmt4k"), \

