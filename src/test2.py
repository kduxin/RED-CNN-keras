 # -*- coding: utf-8 -*- 

from scipy.io import loadmat
import numpy as numpy
import keras
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from train import get_paras
from construction import model_construction_v1

imgid = '0170'
test_x = (pydicom.read_file(f'../data/de/{imgid}.dcm').pixel_array/4000 + 1/4)[np.newaxis,:,:,np.newaxis]
test_y = (pydicom.read_file(f'../data/800/{imgid}.dcm').pixel_array/4000 + 1/4)[np.newaxis,:,:,np.newaxis]

paras = get_paras()
paras['input/h'] = test_x.shape[1]
paras['input/w'] = test_x.shape[2]
model = model_construction_v1(paras)
model.load_weights('../model/v1_20180830_96u_930.weight')

denoise_x = model.predict(test_x)
_, h, w, _ = test_x.shape
contrast = np.zeros([h, 4*w])
contrast[:, :w]    = test_x[0, :, :, 0]
contrast[:, w:2*w] = denoise_x[0, :, :, 0]
contrast[:, 2*w:3*w]  = test_y[0, :, :, 0]
contrast[:, 3*w:]  = denoise_x[0, :, :, 0] - test_y[0, :, :, 0]
plt.imshow(contrast)
plt.show()