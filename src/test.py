 # -*- coding: utf-8 -*- 

from scipy.io import loadmat
import numpy as numpy
import keras
import numpy as np
import matplotlib.pyplot as plt
from util import ImgDataFeeder
from train_v001 import model_construction, get_paras

paras = get_paras()
paras['input/h'] = 273
paras['input/w'] = 273
paras['kernel_size'] = 5
model = model_construction(paras)
# model.load_weights('../model/v001/v1_20180830_96u_930.weight')
model.load_weights('../model/v001/96u.55x55.k5x5.1340.weights')
ID = [3, 11, 19]

# data = loadmat('../data/LowDoseChallenge/test_set.mat')
# test_x = data['test_x'].transpose([3,0,1,2])[ID:ID+1]
# test_y = data['test_y'].transpose([3,0,1,2])[ID:ID+1]
df = ImgDataFeeder('../data/de_800.data.pkl', 
        paras['batch_size'], paras['patch_h'], paras['patch_w'])
test_X, test_Y = df.get_testset()
test_X = test_X[ID]
test_Y = test_Y[ID]

denoise_X = model.predict(test_X)
_, h, w, _ = test_X.shape
contrast = np.zeros([h*len(ID), 4*w])
for i in range(len(ID)):
    contrast[i*h:(i+1)*h, :w]    = test_X[i, :, :, 0]
    contrast[i*h:(i+1)*h, w:2*w] = denoise_X[i, :, :, 0]
    contrast[i*h:(i+1)*h, 2*w:3*w]  = test_Y[i, :, :, 0]
    contrast[i*h:(i+1)*h, 3*w:]  = denoise_X[i, :, :, 0] - test_X[i, :, :, 0]

f = plt.figure(dpi=200)
plt.imshow(contrast, cmap = plt.cm.gray)
plt.show()