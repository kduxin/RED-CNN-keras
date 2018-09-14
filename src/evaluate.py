import os
import pickle as pkl
import numpy as np

from train_v002 import model_construction as mc2
from train_v003 import model_construction as mc3
from train_v004 import model_construction as mc4
from train_v005 import model_construction as mc5
from train_v006 import model_construction as mc6
from train_v007 import model_construction as mc7
from train_v008 import model_construction as mc8
from train_v009 import model_construction as mc9
from train_v010 import model_construction as mc10
from train_v011 import model_construction as mc11
from train_v012 import model_construction as mc12

losses = dict()
for i in range(2, 13):
    folder = '../model/v{:0>3}/'.format(i)
    files = os.listdir(folder)
    files = [file for file in files if file[-4:] == '.pkl']
    hist_file = sorted(files)[-1]
    hist = pkl.loads(open(folder + hist_file, 'rb').read())
    losses['v{:0>3}'.format(i)] = min(hist['loss'])

for key,value in losses.items():
    print(key, value)