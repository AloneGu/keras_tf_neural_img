#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, merge
from keras.models import Model,Sequential
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras.optimizers import Adam, SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave
import time
import numpy as np
import argparse

from keras.callbacks import TensorBoard
from scipy import ndimage

import nets


class FastNeuralStyle(object):
    def __init__(self, style_weight, content_weigth, tv_weight, style_p, img_size):
        self.style_w = style_weight
        self.content_w = content_weigth
        self.tv_w = tv_weight
        self.style_p = style_p
        self.img_size = img_size
        self.img_w, self.img_h = self.img_size

        # build net
        self.net = nets.image_transform_net(self.img_w, self.img_h, self.tv_w)
        self.model = nets.loss_net(self.net.ouput, self.net.input, self.img_w, self.img_h, self.style_p, self.content_w,
                                   self.style_w)

    def train(self, train_dir, nb_epoch, batch):
        learning_rate = 1e-3  # 1e-3
        optimizer = Adam(lr=learning_rate)  # Adam(lr=learning_rate,beta_1=0.99)

        self.model.compile(optimizer, dummy_loss)  # Dummy loss since we are learning from regularizes
        dummy_y = np.zeros((batch, self.img_w, self.img_h, 3))  # Dummy output, not used since we use regularizers to train
        datagen = ImageDataGenerator()

        # start train
        i = 0
        t1 = time.time()
        for x in datagen.flow_from_directory(train_dir, class_mode=None, batch_size=batch,
                                             target_size=(self.img_w, self.img_h), shuffle=False):
            if i > nb_epoch:
                break

            i += 1

            hist = self.model.train_on_batch(x, dummy_y)

            if i % 50 == 0:
                print(hist, (time.time() - t1))
                t1 = time.time()

            # check
            if i % 500 == 0:
                print("epoc: ", i)
                val_x = self.net.predict(x)
                imsave('{}_res.png'.format(i),val_x)
                imsave('{}_res_org.png'.format(i), x)
                self.model.save('/tmp/nn_weights.h5')

if __name__=='__main__':
    nn = FastNeuralStyle(style_weight=4.0,content_weigth=1.0,tv_weight=1e-6,style_p='/home/guwei/style/udnie.jpg',img_size=(256,256))
    nn.train('/home/guwei/nn_style_dir/val2017',2000,1)