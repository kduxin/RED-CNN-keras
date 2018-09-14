import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from util import _load_all_dicom


def Dicom2Dataset(dir_X, dir_Y, save_file, thres_X = 4000, thres_Y = 4000,
                  valid_split = 0.1, test_split = 0.1):
    data_X = _load_all_dicom(dir_X).astype(np.float32)
    data_Y = _load_all_dicom(dir_Y).astype(np.float32)
    data_X = (data_X + 1000) / thres_X
    data_Y = (data_X + 1000) / thres_Y
    data_X[data_X > 1] = 1
    data_X[data_X < 0] = 0
    data_Y[data_Y > 1] = 1
    data_Y[data_Y < 0] = 0
    n_all = data_X.shape[0]
    n_valid = int(n_all * valid_split)
    n_test  = int(n_all * test_split)
    dataset = {
        'train_X' : data_X[:-n_valid-n_test],
        'train_Y' : data_Y[:-n_valid-n_test],
        'valid_X' : data_X[-n_valid-n_test:-n_test],
        'valid_Y' : data_Y[-n_valid-n_test:-n_test],
        'test_X'  : data_X[-n_test:],
        'test_Y'  : data_Y[-n_test:]}
    open(save_file, 'wb').write(pkl.dumps(dataset))