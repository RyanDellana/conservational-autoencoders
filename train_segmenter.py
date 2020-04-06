from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Subtract, Lambda, ReLU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.layers import LeakyReLU, ThresholdedReLU
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, DepthwiseConv2D, ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

import numpy as np
import pickle
import cv2

from models import build_segmenter_model_aspp, build_segmenter_model_aspp_64x64_v2, build_segmenter_model_mobilenetv1_64x64
from models import build_segmenter_model_mobilenetv1_64x64_v2, build_segmenter_model_mobilenetv1_64x64_v3
from models import build_segmenter_model_mobilenetv1_64x64_v4
from models import build_encoder_model_v2, build_decoder_model_v2, build_encoder_model, build_decoder_model
from models import CVAE, VariationalAutoEncoder
from utils import save_images


# Does a parallel shuffle of two numpy arrays.
# Note: Shuffle is not in-place. New arrays are returned.
def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    rng = np.random.RandomState()
    permutation = rng.permutation(a.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def train(save_path):
    batch_size = 64
    # create save directories if they don't exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path+'/model_saves/')
        os.mkdir(save_path+'/images/')

    segmenter, _ = build_segmenter_model_mobilenetv1_64x64_v4(alpha=1, output_cnls=2)
    segmenter.compile(loss      = 'mse', 
                      metrics   = ['accuracy'],
                      optimizer = Adam())

    def kl_divergence(args):
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2.0 * np.pi)
            return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        z_, z_mean_, z_log_var_ = args
        logpz = log_normal_pdf(z_, 0.0, 0.0)
        logqz_x = log_normal_pdf(z_, z_mean_, z_log_var_)
        kl_loss = -tf.reduce_mean(logpz - logqz_x) # complains it can't concatenate over batch dimension.
        #kl_loss = -(logpz - logqz_x)
        return kl_loss

    def mabse_img(args):
        """ Mean absolute error. Used for unbounded functions. """
        x_, x_fake_ = args
        return K.mean(K.abs(x_ - x_fake_), axis=[0,1,2,3], keepdims=False)

    def mse_img(args):
        """ Mean square error. Used for bounded functions. """
        x_, x_fake_ = args
        return K.mean(K.square(x_ - x_fake_), axis=[0,1,2,3], keepdims=False)

    #encoder = build_encoder_model_v2(alpha = 1, in_cnls = 3)
    #decoder = build_decoder_model(alpha = 1, output_cnls = 3)
    # decoder(z0)
    # ^^^ Didn't converge.
    #encoder = build_encoder_model(alpha = 0.5, in_cnls = 3)
    #decoder = build_decoder_model(alpha = 0.5, output_cnls = 3)
    # decoder(z0)
    # ^^^ Didn't converge.
    # encoder = build_encoder_model(alpha = 1, in_cnls = 3)
    # decoder = build_decoder_model(alpha = 1, output_cnls = 3)
    # decoder(z_mean0)
    # ^^^ Converged.
    #encoder = build_encoder_model_v2(alpha = 1, in_cnls = 3)
    #decoder = build_decoder_model_v2(alpha = 1, output_cnls = 3)
    # decoder(z_mean0)
    # ^^^ Didn't converge.
    #encoder = build_encoder_model_v2(alpha = 1, in_cnls = 3)
    #decoder = build_decoder_model(alpha = 1, output_cnls = 3)
    # decoder(z_mean0)
    # ^^^ Didn't converge.

    """
    inputs_vae = Input(shape=(64, 64, 3))
    z, z_mean, z_log_var = encoder(inputs_vae)
    decoder_output = decoder(z)
    loss_re = Lambda(mse_img)([inputs_vae, decoder_output]) # Reconstruction error.
    loss_kl = Lambda(kl_divergence)([z, z_mean, z_log_var]) # KL-divergence loss.
    vae = Model([inputs_vae], [decoder_output, loss_re, loss_kl], name='vae')
    
    vae.add_loss(loss_re + 0.1*loss_kl) # TODO Add loss_kl somehow. TODO
    vae.summary()
    #vae.compile(optimizer = Adadelta(lr=1.0), loss = 'mse')
    #vae.compile(optimizer = Adam(), loss = 'mse')
    vae.compile(optimizer = Adam())
    """

    vae = VariationalAutoEncoder()
    encoder = vae.encoder
    decoder = vae.decoder
    optimizer_vae = Adam(learning_rate=1e-3)

    #vae = CVAE(latent_dim=48)

    # load and normalize dataset
    with open('training_64000_64px.npz', 'rb') as f:
        d = np.load(f)
        x_train, y_train = d['x_train'], d['y_train']
    with open('val_3200_64px.npz', 'rb') as f:
        d = np.load(f)
        x_val, y_val = d['x_train'], d['y_train']
    # remap images from 0.0 -> 1.0 to -1.0 -> 1.0 so they work with tanh activations.
    x_train = x_train * 2.0 - 1.0
    x_val   = x_val * 2.0 - 1.0
    y_train = y_train * 2.0 - 1.0
    y_val   = y_val * 2.0 - 1.0
    print('y_train.shape =', y_train.shape)
    print('y_val.shape =', y_val.shape)
    
    #print('reduce size of training set...')
    x_train = x_train[0:6400, :, :, :]
    y_train = y_train[0:6400, :, :, :]

    print('training...')
    for idx_epoch in range(100):
        print('epoch', idx_epoch)
        x_train, y_train = shuffle_in_unison(x_train, y_train)
        for idx_batch in range(x_train.shape[0] // batch_size):
            print('batch', idx_batch)
            from_, to_ = idx_batch*batch_size, (idx_batch+1)*batch_size
            x_batch, y_batch = x_train[from_:to_], y_train[from_:to_]
            print("x_batch.shape =", x_batch.shape)
            print('segmenter.train_on_batch...')
            #seg_loss  = segmenter.train_on_batch(x=x_batch, y=y_batch)
            #vae_loss  = vae.train_on_batch(x=x_batch)
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch)
                loss_re, loss_kl = vae.losses
                total_loss = loss_re + loss_kl
                grads = tape.gradient(total_loss, vae.trainable_weights)
                optimizer_vae.apply_gradients(zip(grads, vae.trainable_weights))
            #vae.train_on_batch(x=x_batch)
            #print('training loss =', seg_loss)
            #print('training loss vae =', vae_loss)
        #seg_acc, seg_loss = segmenter.evaluate(x=x_val, y=y_val)
        #print('sag_acc, seg_loss =', seg_acc, seg_loss)
        sample_imgs_x = x_val[0:200, :, :, :]
        #sample_imgs_y_ = segmenter.predict(sample_imgs_x)
        #vae_prediction, loss_re, loss_kl = vae.predict(sample_imgs_x)
        vae_prediction = vae(sample_imgs_x)
        vae_prediction = np.array(vae_prediction, dtype=np.float32)
        loss_re, loss_kl = vae.losses
        print('loss_re =', loss_re)
        print('loss_kl =', loss_kl)
        #sample_imgs_y = np.tile(sample_imgs_y_[:, :, :, 0:1], 3) # grab one of the labels and tile it.
        #save_images(path  = save_path+'/images/people'+str(idx_epoch)+'.jpg', 
        #            imgs0 = list(sample_imgs_x), 
        #            imgs1 = list(sample_imgs_y))
        #sample_imgs_y = np.tile(sample_imgs_y_[:, :, :, 1:2], 3) # grab one of the labels and tile it.
        #save_images(path  = save_path+'/images/cars'+str(idx_epoch)+'.jpg', 
        #            imgs0 = list(sample_imgs_x), 
        #            imgs1 = list(sample_imgs_y))
        save_images(path  = save_path+'/images/vae'+str(idx_epoch)+'.jpg',
                    imgs0 = list(sample_imgs_x),
                    imgs1 = list(vae_prediction))


if __name__ == '__main__':
    train(save_path = 'build_segmenter_model_mobilenetv1_64x64_v4_wombat2')


