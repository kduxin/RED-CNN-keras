import numpy as np
from numpy.random import randint
import os
import pydicom
import pickle as pkl

class ImgDataFeeder:
    
    def __init__(self, fpath, batch_size, patch_h, patch_w):
        self.load_data(fpath)
        self.batch_size = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.prepare()
    
    def __iter__(self):
        return self
    
    def load_data(self, fpath):
        path, file = os.path.split(fpath)
        suffix = file[file.rfind('.')+1:]
        if suffix == 'pkl':
            data = pkl.loads(open(fpath, 'rb').read())
            self.train_X = data['train_X']
            self.train_Y = data['train_Y']
            self.valid_X = data['valid_X']
            self.valid_Y = data['valid_Y']
            self.test_X  = data['test_X']
            self.test_Y  = data['test_Y']
    
    def prepare(self):
        self.n_samples, self.h, self.w, _ = self.train_X.shape
        self.patch_h_start = 0
        self.patch_h_end   = self.h - self.patch_h
        self.patch_w_start = 0
        self.patch_w_end   = self.w - self.patch_w

    def __next__(self):
        imgids = randint(0, self.n_samples, [self.batch_size])
        patch_h_starts = randint(self.patch_h_start, self.patch_h_end, [self.batch_size])
        patch_w_starts = randint(self.patch_w_start, self.patch_w_end, [self.batch_size])
        mirror_types = randint(0, 4, [self.batch_size])
        x = []
        y = []
        for i in range(self.batch_size):
            patch_x = self.train_X[imgids[i]:imgids[i]+1, patch_h_starts[i]:patch_h_starts[i]+self.patch_h,
                                patch_w_starts[i]:patch_w_starts[i]+self.patch_w, :]
            patch_y = self.train_Y[imgids[i]:imgids[i]+1, patch_h_starts[i]:patch_h_starts[i]+self.patch_h,
                                patch_w_starts[i]:patch_w_starts[i]+self.patch_w, :]
            if mirror_types[i] == 0:
                x.append(patch_x)
                y.append(patch_y)
            elif mirror_types[i] == 1:
                x.append(patch_x[:,::-1,:,:])
                y.append(patch_y[:,::-1,:,:])
            elif mirror_types[i] == 2:
                x.append(patch_x[:,:,::-1,:])
                y.append(patch_y[:,:,::-1,:])
            else:
                x.append(patch_x[:,::-1,::-1,:])
                y.append(patch_y[:,::-1,::-1,:])
        return np.concatenate(x, axis = 0), np.concatenate(y, axis = 0)

    def get_validset(self):
        return self.valid_X, self.valid_Y
    
    def get_testset(self):
        return self.test_X, self.test_Y


def _load_all_dicom(folder):
    files = os.listdir(folder)
    imgs = []
    i = 0
    for file in sorted(files):
        if file[file.rfind('.')+1:] != 'dcm':
            continue
        i += 1
        imgs.append(pydicom.read_file(os.path.join(folder, file)).pixel_array[np.newaxis,:,:,np.newaxis])
    print("{} dcm files loaded!".format(i))
    imgs = np.concatenate(imgs, axis = 0)
    return imgs


def _append_history(hist, new_hist):
    for key, value in new_hist.items():
        if key not in hist:
            hist[key] = value
        else:
            hist[key] += value
    return hist

def _check_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    print(f'Dir created: {dirname}')