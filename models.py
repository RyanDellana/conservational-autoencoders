from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Subtract, Lambda, ReLU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.layers import LeakyReLU, ThresholdedReLU

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np


def sampling(args): # TODO Need to attribute this properly. TODO
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


def build_encoder_model(filters      = 32, 
                        latent_dims  = 48,
                        z_activation = 'leaky_relu',
                        in_cnls      = 3):
    inputs = Input(shape=(64, 64, in_cnls))
    x  = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(inputs)
    x  = LeakyReLU(alpha=0.2)(x)
    x  = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    layers = [(filters*2, (2,2)),
              (filters*2, (1,1)), 
              (filters*4, (2,2)), 
              (filters*4, (1,1)), 
              (filters*4, (1,1))]
    for n_filters, strides in layers:                                                                                       
        x = Conv2D(filters     = n_filters,
                   kernel_size = (5,5), 
                   activation  = None, 
                   strides     = strides, 
                   padding     = 'same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    z_mean    = Dense(latent_dims, name='z_mean_encode')(x)
    z_log_var = Dense(latent_dims, name='z_log_var_encode')(x)
    z         = Lambda(sampling, output_shape=(latent_dims,), name='z_encode')([z_mean, z_log_var])
    if z_activation in ['relu', 'sigmoid', 'tanh']:
        z = Activation(z_activation)(z)
    elif z_activation == 'leaky_relu':
        z = LeakyReLU(alpha=0.2)(z)
    elif z_activation == 'bounded_relu':
        z = ReLU(max_value=1.0)(z)
    encoder = Model([inputs], [z, z_mean, z_log_var], name='encoder')
    encoder.summary()
    return encoder


def build_decoder_model(filters=32, 
                        latent_dims=48, 
                        output_cnls=3,
                        out_act='tanh'):
    z_input = Input(shape=(latent_dims,))
    x = Dense(16*16*filters*4, activation=None)(z_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((16, 16, filters*4))(x)
    layers = [(filters*4, (1,1)),
              (filters*4, (1,1)),
              (filters*4, (2,2)),
              (filters*2, (1,1)), 
              (filters*2, (2,2)), 
              (filters,   (1,1)), 
              (filters,   (1,1))]
    for n_filters, strides in layers:
        x = Conv2DTranspose(filters=n_filters,
                            kernel_size=(5,5),
                            activation=None,
                            strides=strides,
                            padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    output = Conv2DTranspose(filters     = output_cnls, 
                             kernel_size = (5,5), 
                             activation  = out_act, 
                             strides     = (1,1), 
                             padding     = 'same')(x)
    decoder = Model(z_input, output)
    decoder.summary()
    return decoder


def build_vae_model(filters      = 32, 
                    latent_dims  = 48, 
                    z_activation = 'leaky_relu',
                    input_cnls   = 3,
                    output_cnls  = 3):
    encoder = build_encoder_model(filters      = filters, 
                                  latent_dims  = latent_dims, 
                                  z_activation = z_activation,
                                  in_cnls      = input_cnls)
    decoder = build_decoder_model(filters      = filters, 
                                  latent_dims  = latent_dims,
                                  output_cnls  = output_cnls,
                                  out_act      = 'tanh')
    inputs = Input(shape=(64,64,3))
    z, z_mean, z_log_var = encoder(input_img)
    decoder_output = decoder(z)
    vae = Model([inputs], [decoder_output, z_mean, z_log_var], name='vae')
    return vae, encoder, decoder


def build_segmenter_model(filters=32, 
                          output_cnls=2):
    inputs = Input(shape=(64,64,3))
    x = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    t1 = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(2,2), padding='same')(t1)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters*2, kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=filters, kernel_size=(5,5), activation=None, strides=(4,4), padding='same')(x)
    t7 = LeakyReLU(alpha=0.2)(x)
    tcat = Concatenate(axis=-1)([t1, t7])
    output = Conv2D(filters=output_cnls, kernel_size=(5,5), activation='tanh', strides=(1,1), padding='same')(tcat)
    model_skip     = Model([inputs], [output, t1, t7])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip    


def build_composite_model(path_pretrained_segmenter    = None,
                          path_pretrained_encoder      = None,
                          path_pretrained_decoder      = None,
                          w_loss_feature_conserv       = 0.25,
                          w_loss_deep_feature_conserv  = 0.25,
                          w_loss_output_conserv_person = 0.0,
                          w_loss_output_conserv_car    = 0.0,
                          w_loss_fakeness              = 0.25,
                          w_loss_kl_divergence         = 0.25,
                          filters                      = 32,
                          encode_segmenter_output      = True,
                          encode_segmenter_skips       = False):
    # ---- Define and load segmenter model ----
    segmenter_training, segmenter = build_segmenter_model(filters = filters, output_cnls = 3)
    if path_pretrained_segmenter:
        segmenter.load_weights(path_pretrained_segmenter)

    segmenter.compile(loss = 'mse', optimizer = Adam(0.0001, 0.5))
    segmenter_training.compile(loss      = 'mse', 
                               metrics   = ['accuracy'],
                               optimizer = Adam(0.0001, 0.5))

    segmenter.trainable           = False
    segmenter_training.trainable  = False
    # ---- Define and load autoencoder ----
    in_cnls = 3
    if encode_segmenter_output:
        in_cnls += 2
    if encode_segmenter_skips:
        in_cnls += filters*2
    encoder = build_encoder_model(filters      = filters, 
                                  latent_dims  = 48, 
                                  z_activation = 'leaky_relu',
                                  in_cnls      = in_cnls)
    decoder = build_decoder_model(filters      = filters, 
                                  latent_dims  = 48,
                                  output_cnls  = 3,
                                  out_act      = 'tanh')
    inputs_vae = Input(shape=(64, 64, in_cnls))
    z0, z_mean0, z_log_var0 = encoder(inputs_vae)
    decoder_output = decoder(z0)
    vae = Model([inputs_vae], [decoder_output, z0, z_mean0, z_log_var0], name='vae')
    if path_pretrained_encoder:
        encoder.load_weights(path_pretrained_encoder)
    if path_pretrained_decoder:
        decoder.load_weights(path_pretrained_decoder)
    # ---- Define composite model ----
    x = Input(shape=(64, 64, 3))
    (seg_x, features_x, deep_features_x) = segmenter(x)
    seg_x = Lambda(lambda t: (t + 1.0) / 2.0)(seg_x) # Remap from -1.0 -> 1.0 to 0.0 -> 1.0
    seg_x = Lambda(lambda t: t[:,:,:,0:2])(seg_x) # Remove the "fake-channel"
    if encode_segmenter_output and encode_segmenter_skips:
        vae_input = Concatenate(axis=-1)([x, features_x, deep_features_x, seg_x])
    elif encode_segmenter_output:
        vae_input = Concatenate(axis=-1)([x, seg_x])
    elif encode_segmenter_skips:
        vae_input = Concatenate(axis=-1)([x, features_x, deep_features_x])
    else:
        vae_input = x
    x_fake, z1, z_mean1, z_log_var1 = vae(vae_input)
    (seg_x_fake, features_x_fake, deep_features_x_fake) = segmenter(x_fake)
    seg_x_fake = Lambda(lambda t: (t + 1.0) / 2.0)(seg_x_fake) # Remap from -1.0 -> 1.0 to 0.0 -> 1.0
    fake_cnl   = Lambda(lambda t: t[:,:,:,2:3])(seg_x_fake) # Separate out fake-channel.
    seg_x_fake = Lambda(lambda t: t[:,:,:,0:2])(seg_x_fake) # Remove the "fake-channel"

    """
    # http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    def kl_divergence(args):
        z_mean_, z_log_var_ = args
        kl_loss = 1 + z_log_var_ - tf.keras.backend.square(z_mean_) - tf.keras.backend.exp(z_log_var_)
        #kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        #kl_loss *= -0.5
        #kl_loss = tf.keras.backend.mean(kl_loss)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=[1], keepdims=False)
        kl_loss *= -0.5
        kl_loss = tf.keras.backend.mean(kl_loss, keepdims=True)
        return kl_loss
    """

    def kl_divergence(args):
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2.0 * np.pi)
            return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        z_, z_mean_, z_log_var_ = args
        logpz = log_normal_pdf(z_, 0.0, 0.0)
        logqz_x = log_normal_pdf(z_, z_mean_, z_log_var_)
        #kl_loss = -tf.reduce_mean(logpz - logqz_x) # complains it can't concatenate over batch dimension.
        kl_loss = -(logpz - logqz_x)
        return kl_loss

    def mse_img(args):
        """ Mean square error. Used for bounded functions. """
        x_, x_fake_ = args
        return tf.keras.backend.mean(tf.keras.backend.square(x_ - x_fake_), axis=[1,2,3], keepdims=False)

    def mabse_img(args):
        """ Mean absolute error. Used for unbounded functions. """
        x_, x_fake_ = args
        return tf.keras.backend.mean(tf.keras.backend.abs(x_ - x_fake_), axis=[1,2,3], keepdims=False)

    def mse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return tf.keras.backend.mean(tf.keras.backend.square((x_ - x_fake_) * attn_map), axis=[1,2,3], keepdims=False)

    def mabse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return tf.keras.backend.mean(tf.keras.backend.abs((x_ - x_fake_) * attn_map), axis=[1,2,3], keepdims=False)

    def intersection_over_union(args): # TODO Is there a better way to do axis=[1,2,3]?
        y_true, y_pred = args
        return tf.keras.backend.sum(tf.keras.backend.minimum(y_true, y_pred), axis=[1,2,3]) / tf.keras.backend.sum(tf.keras.backend.maximum(y_true, y_pred), axis=[1,2,3])

    seg_g                      = Input(shape=(64, 64, 3)) # Segmentation ground truth.
    seg_g_wout_fake_cnl        = Lambda(lambda t: t[:,:,:,0:2])(seg_g) # Remove "fake" channel.
    seg_g_person               = Lambda(lambda t: t[:,:,:,0:1])(seg_g)
    seg_g_car                  = Lambda(lambda t: t[:,:,:,1:2])(seg_g)
    seg_x_person               = Lambda(lambda t: t[:,:,:,0:1])(seg_x)
    seg_x_car                  = Lambda(lambda t: t[:,:,:,1:2])(seg_x)
    seg_x_fake_person          = Lambda(lambda t: t[:,:,:,0:1])(seg_x_fake)
    seg_x_fake_car             = Lambda(lambda t: t[:,:,:,1:2])(seg_x_fake)
    loss_re                    = Lambda(mabse_img)([x, x_fake]) # Reconstruction error.
    loss_ammre_person          = Lambda(mabse_amm_img)([x, x_fake, seg_x_person])
    loss_ammre_car             = Lambda(mabse_amm_img)([x, x_fake, seg_x_car])
    loss_output_conserv_person = Lambda(mse_img)([seg_x_person, seg_x_fake_person]) # Output conservation error person.
    loss_output_conserv_car    = Lambda(mse_img)([seg_x_car, seg_x_fake_car]) # Output conservation error car.
    loss_feature_conserv       = Lambda(mabse_img)([features_x, features_x_fake]) # Feature conservation error.
    loss_deep_feature_conserv  = Lambda(mabse_img)([deep_features_x, deep_features_x_fake]) # Deep feature conservation error.
    loss_fakeness              = Lambda(lambda t: tf.keras.backend.mean(t, axis=[1,2,3], keepdims=False))(fake_cnl)
    # ^^^ should these be mean-square-error or mean-absolute-error, since they're unbounded functions. TODO
    # ^^^ Also note that we are not comparing apples-to-apples so, output_conserv losses can't be used with feature_conserv losses. TODO
    loss_kl                    = Lambda(kl_divergence)([z1, z_mean1, z_log_var1]) # KL-divergence loss.
    iou_seg_x_seg_x_fake       = Lambda(intersection_over_union)([seg_x, seg_x_fake])
    iou_seg_g_seg_x            = Lambda(intersection_over_union)([seg_g_wout_fake_cnl, seg_x])
    iou_seg_g_seg_x_fake       = Lambda(intersection_over_union)([seg_g_wout_fake_cnl, seg_x_fake])
    loss_composite             = tf.keras.backend.mean(loss_output_conserv_person * w_loss_output_conserv_person \
                                                    + loss_output_conserv_car     * w_loss_output_conserv_car \
                                                    + loss_feature_conserv        * w_loss_feature_conserv \
                                                    + loss_deep_feature_conserv   * w_loss_deep_feature_conserv \
                                                    + loss_fakeness               * w_loss_fakeness \
                                                    + loss_kl                     * w_loss_kl_divergence) # TODO Why mean?
    composite_outputs = [x_fake, 
                         seg_x,
                         seg_x_fake,
                         loss_re,
                         loss_feature_conserv, 
                         loss_deep_feature_conserv, 
                         loss_output_conserv_person,
                         loss_output_conserv_car,
                         loss_fakeness,
                         loss_kl, 
                         iou_seg_x_seg_x_fake,
                         iou_seg_g_seg_x,
                         iou_seg_g_seg_x_fake]
    composite = Model([x, seg_g], composite_outputs)
    composite.add_loss(loss_composite)
    composite.summary()
    composite.compile(optimizer = Adam(0.0001, 0.5))
    return segmenter_training, vae, encoder, decoder, composite
    

