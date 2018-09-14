 # -*- coding: utf-8 -*- 

from scipy.io import loadmat
from numpy.random import randint
import numpy as np
from keras.models import load_model
from keras.initializers import RandomNormal
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.layers import Input, Conv2D, Deconv2D, ReLU, Add, BatchNormalization
from keras.activations import relu
from keras.models import Model
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
from keras.initializers import Orthogonal
from keras.losses import mse
import keras.backend as K
import datetime as dt
import os
import pydicom
import pickle as pkl
from util import _check_dir, _append_history, _load_all_dicom, ImgDataFeeder

def model_construction(paras):
    
    inputs = Input(shape=(paras['input/h'], paras['input/w'], 1))
    conv1  = Conv2D(filters = paras['conv1/filters'], kernel_size = paras['kernel_size'], 
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(inputs)
    if paras['use_batchnorm']:
        conv1 = BatchNormalization()(conv1)

    conv2  = Conv2D(filters = paras['conv2/filters'], kernel_size = paras['kernel_size'], 
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(conv1)
    if paras['use_batchnorm']:
        conv2 = BatchNormalization()(conv2)

    conv3  = Conv2D(filters = paras['conv3/filters'], kernel_size = paras['kernel_size'], 
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(conv2)
    if paras['use_batchnorm']:
        conv3 = BatchNormalization()(conv3)

    conv4  = Conv2D(filters = paras['conv4/filters'], kernel_size = paras['kernel_size'], 
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(conv3)
    if paras['use_batchnorm']:
        conv4 = BatchNormalization()(conv4)

    conv5  = Conv2D(filters = paras['conv5/filters'], kernel_size = paras['kernel_size'], 
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(conv4)
    if paras['use_batchnorm']:
        conv5 = BatchNormalization()(conv5)

    dconv5 = Deconv2D(filters = paras['dconv5/filters'], kernel_size = paras['kernel_size'],
                padding = paras['padding'],
                kernel_initializer = paras['kernel_initializer'])(conv5)
    drelu5 = ReLU()(Add()([dconv5, conv4]))
    if paras['use_batchnorm']:
        drelu5 = BatchNormalization()(drelu5)

    dconv4 = Deconv2D(filters = paras['dconv4/filters'], kernel_size = paras['kernel_size'],
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(drelu5)
    if paras['use_batchnorm']:
        dconv4 = BatchNormalization()(dconv4)

    dconv3 = Deconv2D(filters = paras['dconv3/filters'], kernel_size = paras['kernel_size'],
                padding = paras['padding'],
                kernel_initializer = paras['kernel_initializer'])(dconv4)
    drelu3 = ReLU()(Add()([dconv3, conv2]))
    if paras['use_batchnorm']:
        drelu3 = BatchNormalization()(drelu3)

    dconv2 = Deconv2D(filters = paras['dconv2/filters'], kernel_size = paras['kernel_size'],
                padding = paras['padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(drelu3)
    if paras['use_batchnorm']:
        dconv2 = BatchNormalization()(dconv2)

    dconv1 = Deconv2D(filters = paras['dconv1/filters'], kernel_size = paras['kernel_size'],
                padding = paras['padding'],
                kernel_initializer = paras['kernel_initializer'])(dconv2)
    outputs = ReLU()(Add()([dconv1, inputs]))

    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(loss = 'mse', optimizer = Adam(lr = paras['lr/base'],
                decay = paras['lr/decay']))
    return model

def get_paras():
    paras = dict()
    paras['input/h'] = 55
    paras['input/w'] = 55
    paras['use_batchnorm'] = False
    n_units = 16
    paras['conv1/filters'] = n_units
    paras['conv2/filters'] = n_units
    paras['conv3/filters'] = n_units
    paras['conv4/filters'] = n_units
    paras['conv5/filters'] = n_units
    paras['dconv5/filters'] = n_units
    paras['dconv4/filters'] = n_units
    paras['dconv3/filters'] = n_units
    paras['dconv2/filters'] = n_units
    paras['dconv1/filters'] = 1
    paras['kernel_size'] = 5
    paras['padding'] = 'same'

    paras['kernel_initializer'] = RandomNormal()

    paras['lr/base']  = 1e-4
    paras['lr/decay'] = 1e-4

    paras['batch_size'] = 128
    paras['patch_h'] = paras['input/h']
    paras['patch_w'] = paras['input/w']
    paras['epochs']  = 10
    paras['steps_per_epoch'] = 100
    paras['end_time'] = '20180913 10:00:00'
    return paras

class print_lr(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

def train_model(model, paras, start_epoch = 0):
    version = '008'
    _check_dir(f'../model/v{version}')
    prefix = '{}u.{}x{}.k{}x{}'.format(
                        paras['conv1/filters'], paras['input/h'], paras['input/w'],
                        paras['kernel_size'], paras['kernel_size'])
    plot_model(model, '../model/v{}/{}.png'.format(version, prefix))
    df = ImgDataFeeder('../data/de_800.data.pkl', paras['batch_size'], paras['patch_h'], paras['patch_w'])
    epoch = start_epoch
    train_hist = {}
    model.save('../model/v{}/{}.{:0>4}.model'.format(version, prefix, epoch))
    open('../model/v{}/{}.{:0>4}.history.pkl'.format(version, prefix, epoch), 'wb').write(pkl.dumps(train_hist))
    print(f"Current epoch: {epoch}. Current lr: {model.optimizer.lr}. Model has been saved.")
    while dt.datetime.now() < dt.datetime.strptime(paras['end_time'], 
                                                    '%Y%m%d %H:%M:%S'):
        hist = model.fit_generator(df, epochs = paras['epochs'],
                                steps_per_epoch = paras['steps_per_epoch'], 
                                validation_steps = 5)
        train_hist = _append_history(train_hist, hist.history)
        epoch += paras['epochs']
        model.save_weights('../model/v{}/{}.{:0>4}.weights'.format(version, prefix, epoch))
        open('../model/v{}/{}.{:0>4}.history.pkl'.format(version, prefix, epoch), 'wb').write(pkl.dumps(train_hist))
        print(f"Current epoch: {epoch}. Current lr: {model.optimizer.lr}. Model has been saved.")
    return model, train_hist

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    paras = get_paras()
    paras['lr/base'] = 1e-4
    paras['end_time'] = '20180914 20:00:00'
    model = model_construction(paras)
    model = train_model(model, paras, 0)