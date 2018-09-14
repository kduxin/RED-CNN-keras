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
from util import _check_dir, _append_history, ImgDataFeeder, _load_all_dicom

def model_construction(paras):
    
    inputs = Input(shape=(paras['input/h'], paras['input/w'], 1))
    # branch 1
    b1_conv1  = Conv2D(filters = paras['b1/conv1/filters'], kernel_size = paras['b1/kernel_size'], 
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(inputs)

    b1_conv2  = Conv2D(filters = paras['b1/conv2/filters'], kernel_size = paras['b1/kernel_size'], 
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_conv1)

    b1_conv3  = Conv2D(filters = paras['b1/conv3/filters'], kernel_size = paras['b1/kernel_size'], 
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_conv2)

    b1_conv4  = Conv2D(filters = paras['b1/conv4/filters'], kernel_size = paras['b1/kernel_size'], 
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_conv3)

    b1_conv5  = Conv2D(filters = paras['b1/conv5/filters'], kernel_size = paras['b1/kernel_size'], 
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_conv4)

    b1_dconv5 = Deconv2D(filters = paras['b1/dconv5/filters'], kernel_size = paras['b1/kernel_size'],
                padding = paras['b1/padding'],
                kernel_initializer = paras['kernel_initializer'])(b1_conv5)
    b1_drelu5 = ReLU()(Add()([b1_dconv5, b1_conv4]))

    b1_dconv4 = Deconv2D(filters = paras['b1/dconv4/filters'], kernel_size = paras['b1/kernel_size'],
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_drelu5)

    b1_dconv3 = Deconv2D(filters = paras['b1/dconv3/filters'], kernel_size = paras['b1/kernel_size'],
                padding = paras['b1/padding'],
                kernel_initializer = paras['kernel_initializer'])(b1_dconv4)
    b1_drelu3 = ReLU()(Add()([b1_dconv3, b1_conv2]))

    b1_dconv2 = Deconv2D(filters = paras['b1/dconv2/filters'], kernel_size = paras['b1/kernel_size'],
                padding = paras['b1/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b1_drelu3)

    b1_dconv1 = Deconv2D(filters = paras['b1/dconv1/filters'], kernel_size = paras['b1/kernel_size'],
                padding = paras['b1/padding'],
                kernel_initializer = paras['kernel_initializer'])(b1_dconv2)
    
    # branch2
    b2_conv1  = Conv2D(filters = paras['b2/conv1/filters'], kernel_size = paras['b2/kernel_size'], 
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(inputs)

    b2_conv2  = Conv2D(filters = paras['b2/conv2/filters'], kernel_size = paras['b2/kernel_size'], 
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_conv1)

    b2_conv3  = Conv2D(filters = paras['b2/conv3/filters'], kernel_size = paras['b2/kernel_size'], 
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_conv2)

    b2_conv4  = Conv2D(filters = paras['b2/conv4/filters'], kernel_size = paras['b2/kernel_size'], 
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_conv3)

    b2_conv5  = Conv2D(filters = paras['b2/conv5/filters'], kernel_size = paras['b2/kernel_size'], 
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_conv4)

    b2_dconv5 = Deconv2D(filters = paras['b2/dconv5/filters'], kernel_size = paras['b2/kernel_size'],
                padding = paras['b2/padding'],
                kernel_initializer = paras['kernel_initializer'])(b2_conv5)
    b2_drelu5 = ReLU()(Add()([b2_dconv5, b2_conv4]))

    b2_dconv4 = Deconv2D(filters = paras['b2/dconv4/filters'], kernel_size = paras['b2/kernel_size'],
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_drelu5)

    b2_dconv3 = Deconv2D(filters = paras['b2/dconv3/filters'], kernel_size = paras['b2/kernel_size'],
                padding = paras['b2/padding'],
                kernel_initializer = paras['kernel_initializer'])(b2_dconv4)
    b2_drelu3 = ReLU()(Add()([b2_dconv3, b2_conv2]))

    b2_dconv2 = Deconv2D(filters = paras['b2/dconv2/filters'], kernel_size = paras['b2/kernel_size'],
                padding = paras['b2/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b2_drelu3)

    b2_dconv1 = Deconv2D(filters = paras['b2/dconv1/filters'], kernel_size = paras['b2/kernel_size'],
                padding = paras['b2/padding'],
                kernel_initializer = paras['kernel_initializer'])(b2_dconv2)


    # branch3
    b3_conv1  = Conv2D(filters = paras['b3/conv1/filters'], kernel_size = paras['b3/kernel_size'], 
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(inputs)

    b3_conv2  = Conv2D(filters = paras['b3/conv2/filters'], kernel_size = paras['b3/kernel_size'], 
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_conv1)

    b3_conv3  = Conv2D(filters = paras['b3/conv3/filters'], kernel_size = paras['b3/kernel_size'], 
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_conv2)

    b3_conv4  = Conv2D(filters = paras['b3/conv4/filters'], kernel_size = paras['b3/kernel_size'], 
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_conv3)

    b3_conv5  = Conv2D(filters = paras['b3/conv5/filters'], kernel_size = paras['b3/kernel_size'], 
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_conv4)

    b3_dconv5 = Deconv2D(filters = paras['b3/dconv5/filters'], kernel_size = paras['b3/kernel_size'],
                padding = paras['b3/padding'],
                kernel_initializer = paras['kernel_initializer'])(b3_conv5)
    b3_drelu5 = ReLU()(Add()([b3_dconv5, b3_conv4]))

    b3_dconv4 = Deconv2D(filters = paras['b3/dconv4/filters'], kernel_size = paras['b3/kernel_size'],
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_drelu5)

    b3_dconv3 = Deconv2D(filters = paras['b3/dconv3/filters'], kernel_size = paras['b3/kernel_size'],
                padding = paras['b3/padding'],
                kernel_initializer = paras['kernel_initializer'])(b3_dconv4)
    b3_drelu3 = ReLU()(Add()([b3_dconv3, b3_conv2]))

    b3_dconv2 = Deconv2D(filters = paras['b3/dconv2/filters'], kernel_size = paras['b3/kernel_size'],
                padding = paras['b3/padding'], activation = 'relu',
                kernel_initializer = paras['kernel_initializer'])(b3_drelu3)

    b3_dconv1 = Deconv2D(filters = paras['b3/dconv1/filters'], kernel_size = paras['b3/kernel_size'],
                padding = paras['b3/padding'],
                kernel_initializer = paras['kernel_initializer'])(b3_dconv2)
    
    outputs = ReLU()(Add()([b1_dconv1, b2_dconv1, b3_dconv1, inputs]))

    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(loss = 'mse', optimizer = Adam(lr = paras['lr/base'],
                decay = paras['lr/decay']))
    return model

def get_paras():
    paras = dict()
    paras['input/h'] = 55
    paras['input/w'] = 55
    paras['use_batchnorm'] = False

    n_units_b1 = 32
    paras['b1/conv1/filters'] = n_units_b1
    paras['b1/conv2/filters'] = n_units_b1
    paras['b1/conv3/filters'] = n_units_b1
    paras['b1/conv4/filters'] = n_units_b1
    paras['b1/conv5/filters'] = n_units_b1
    paras['b1/dconv5/filters'] = n_units_b1
    paras['b1/dconv4/filters'] = n_units_b1
    paras['b1/dconv3/filters'] = n_units_b1
    paras['b1/dconv2/filters'] = n_units_b1
    paras['b1/dconv1/filters'] = 1
    paras['b1/kernel_size'] = 5
    paras['b1/padding'] = 'same'

    n_units_b2 = 32
    paras['b2/conv1/filters'] = n_units_b2
    paras['b2/conv2/filters'] = n_units_b2
    paras['b2/conv3/filters'] = n_units_b2
    paras['b2/conv4/filters'] = n_units_b2
    paras['b2/conv5/filters'] = n_units_b2
    paras['b2/dconv5/filters'] = n_units_b2
    paras['b2/dconv4/filters'] = n_units_b2
    paras['b2/dconv3/filters'] = n_units_b2
    paras['b2/dconv2/filters'] = n_units_b2
    paras['b2/dconv1/filters'] = 1
    paras['b2/kernel_size'] = 3
    paras['b2/padding'] = 'same'

    n_units_b3 = 16
    paras['b3/conv1/filters'] = n_units_b3
    paras['b3/conv2/filters'] = n_units_b3
    paras['b3/conv3/filters'] = n_units_b3
    paras['b3/conv4/filters'] = n_units_b3
    paras['b3/conv5/filters'] = n_units_b3
    paras['b3/dconv5/filters'] = n_units_b3
    paras['b3/dconv4/filters'] = n_units_b3
    paras['b3/dconv3/filters'] = n_units_b3
    paras['b3/dconv2/filters'] = n_units_b3
    paras['b3/dconv1/filters'] = 1
    paras['b3/kernel_size'] = 7
    paras['b3/padding'] = 'same'

    paras['kernel_initializer'] = RandomNormal()

    paras['lr/base']  = 1e-4
    paras['lr/decay'] = 1e-4

    paras['batch_size'] = 128
    paras['patch_h'] = paras['input/h']
    paras['patch_w'] = paras['input/w']
    paras['epochs']  = 10
    paras['steps_per_epoch'] = 100
    paras['end_time'] = '20180914 20:00:00'
    return paras

class print_lr(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

def train_model(model, paras, start_epoch = 0, train_hist = {}):
    version = '010'
    _check_dir('../model/v{}'.format(version))
    prefix = '{}x{}'.format(
                        paras['input/h'], paras['input/w'])
    plot_model(model, '../model/v{}/{}.png'.format(version, prefix))
    df = ImgDataFeeder('../data/de_800.data1-3.pkl', paras['batch_size'], paras['patch_h'], paras['patch_w'])
    epoch = start_epoch
    model.save('../model/v{}/{}.{:0>4}.model'.format(version, prefix, epoch))
    open('../model/v{}/{}.{:0>4}.history.pkl'.format(version, prefix, epoch), 'wb').write(pkl.dumps(train_hist))
    print("Current epoch: {}. Current lr: {}. "
          "Model has been saved.".format(epoch, model.optimizer.lr))
    while dt.datetime.now() < dt.datetime.strptime(paras['end_time'], 
                                                    '%Y%m%d %H:%M:%S'):
        hist = model.fit_generator(df, epochs = paras['epochs'],
                                steps_per_epoch = paras['steps_per_epoch'], 
                                validation_steps = 5)
        train_hist = _append_history(train_hist, hist.history)
        epoch += paras['epochs']
        model.save_weights('../model/v{}/{}.{:0>4}.weights'.format(version, prefix, epoch))
        open('../model/v{}/{}.{:0>4}.history.pkl'.format(version, prefix, epoch), 'wb').write(pkl.dumps(train_hist))
        print("Current epoch: {}. Current lr: {}. "
              "Model has been saved.".format(
                        epoch, model.optimizer.lr))
    return model, train_hist

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    paras = get_paras()
    paras['lr/base'] = 1e-5
    paras['end_time'] = '20180915 12:00:00'
    model = model_construction(paras)
    model.load_weights('../model/v010/55x55.0560.weights')
    train_hist = pkl.loads(open('../model/v010/55x55.0560.history.pkl', 'rb').read())
    model = train_model(model, paras, 560, train_hist)