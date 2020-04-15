from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Flatten, Subtract, Lambda, ReLU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.layers import LeakyReLU, ThresholdedReLU
from tensorflow.keras.layers import Add, Dropout, BatchNormalization, DepthwiseConv2D, ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam, Adadelta

from tensorflow.keras import backend as K

import numpy as np


class CVAE(Model):
  
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        inputs = Input((64, 64, 3))
        x = Conv2D(filters = 32, kernel_size = 3, strides = (2, 2), activation='relu')(inputs) # (32, 32, 32)
        x = Conv2D(filters = 64, kernel_size = 3, strides = (2, 2), activation='relu')(x)      # (16, 16, 64)
        x = Conv2D(filters = 128, kernel_size = 3, strides = (2, 2), activation='relu')(x)     # (8, 8, 128)
        x = Conv2D(filters = 128, kernel_size = 3, strides = (2, 2), activation='relu')(x)     # (4, 4, 128)
        x = Flatten()(x)
        output = Dense(latent_dim + latent_dim)(x)
        self.inference_net = Model([inputs], [output], name='inference_net')
        inputs = Input((latent_dim,))
        x = Dense(units=4*4*128, activation='relu')(inputs)
        x = Reshape((4, 4, 128))(x)
        x = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='relu')(x) # (8, 8, 128)
        x = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu')(x)  # (16, 16, 64)
        x = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu')(x)  # (32, 32, 32)
        x = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu')(x)  # (64, 64, 32)
        output = Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', activation='tanh')(x) # (64, 64, 3)
        self.generative_net = Model([inputs], [output], name='generative_net')
        self.optimizer = Adam()

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        x_sig = (x + 1.0) / 2.0
        x_logit_sig = (x_logit + 1.0) / 2.0
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit_sig, labels=x_sig)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def predict(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        return x_logit

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def train_on_batch(self, x):
        self.compute_apply_gradients(x)


# ==============================================

class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch   = tf.shape(z_mean)[0]
        dim     = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return z_mean + (tf.exp(0.5 * z_log_var) * epsilon)


class Encoder(Model):

    def __init__(self, alpha=1.0, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv0  = Conv2D(filters = int(32*alpha),  kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv1  = Conv2D(filters = int(64*alpha),  kernel_size = 5, strides = 2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv3  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv4  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv5  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.flat   = Flatten()
        self.dense0 = Dense(int(128*alpha), activation = LeakyReLU(alpha=0.2))
        self.dense1 = Dense(48, activation = None)

    def call(self, inputs):
        x = self.conv0(inputs) # (64, 64, 32)   CRF = 5
        x = self.conv1(x)      # (64, 64, 64)   CRF = 9
        x = self.conv2(x)      # (32, 32, 128)  CRF = 13
        x = self.conv3(x)      # (32, 32, 128)  CRF = 21
        x = self.conv4(x)      # (16, 16, 128)  CRF = 29
        x = self.conv5(x)      # (16, 16, 128)  CRF = 45
        x = self.flat(x)       # (32768, )
        x = self.dense0(x)     # (128, )
        z = self.dense1(x)
        return z


class Encoder_VAE(Model):

    def __init__(self, alpha=1.0, name='encoder', **kwargs):
        super(Encoder_VAE, self).__init__(name=name, **kwargs)
        self.conv0  = Conv2D(filters = int(32*alpha),  kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv1  = Conv2D(filters = int(64*alpha),  kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2)) 
        self.conv2  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv3  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv4  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv5  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv6  = Conv2D(filters = int(128*alpha), kernel_size = 5, strides = 1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.flat   = Flatten()
        self.dense0 = Dense(int(128*alpha), activation = LeakyReLU(alpha=0.2))
        self.dense_mean    = Dense(48, activation = None)
        self.dense_log_var = Dense(48, activation = None)
        self.sampling     = Sampling()

    def call(self, inputs):
        x = self.conv0(inputs) # (64, 64, 32)   CRF = 5
        x = self.conv1(x)      # (64, 64, 64)   CRF = 9
        x = self.conv2(x)      # (32, 32, 128)  CRF = 13
        x = self.conv3(x)      # (32, 32, 128)  CRF = 21
        x = self.conv4(x)      # (16, 16, 128)  CRF = 29
        x = self.conv5(x)      # (16, 16, 128)  CRF = 45
        x = self.conv6(x)      # (16, 16, 128)  CRF = 61
        x = self.flat(x)       # (32768, )
        x = self.dense0(x)     # (128, )
        z_mean    = self.dense_mean(x)    # (48, )
        z_log_var = self.dense_log_var(x) # (48, )
        #z_mean    = K.clip(min_value=0.0, max_value=100.0, x=z_mean)
        #z_log_var = K.clip(min_value=0.0, max_value=100.0, x=z_log_var)
        z         = self.sampling((z_mean, z_log_var)) # (48, )
        return z_mean, z_log_var, z


class Decoder(Model):

    def __init__(self, alpha=1.0, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense0  = Dense(int(128*alpha), activation = LeakyReLU(alpha=0.2))
        self.dense1  = Dense(int(128*16*16*alpha), activation = LeakyReLU(alpha=0.2))
        self.reshape = Reshape((16, 16, 128))
        self.conv2dtrans0    = Conv2DTranspose(int(128*alpha), kernel_size = 5, strides=1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans1    = Conv2DTranspose(int(128*alpha), kernel_size = 5, strides=1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans2    = Conv2DTranspose(int(128*alpha), kernel_size = 5, strides=2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans3    = Conv2DTranspose(int(128*alpha), kernel_size = 5, strides=1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans4    = Conv2DTranspose(int(64*alpha),  kernel_size = 5, strides=2, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans5    = Conv2DTranspose(int(32*alpha),  kernel_size = 5, strides=1, padding='same', activation = LeakyReLU(alpha=0.2))
        self.conv2dtrans_out = Conv2DTranspose(3,              kernel_size = 5, strides=1, padding='same', activation = 'tanh')

    def call(self, inputs):
        x = self.dense0(inputs)
        x = self.dense1(x)
        x = self.reshape(x)      # (4, 4, 3)
        x = self.conv2dtrans0(x)      # (16, 16, 128)
        x = self.conv2dtrans1(x)      # (16, 16, 128)
        x = self.conv2dtrans2(x)      # (16, 16, 128)
        x = self.conv2dtrans3(x)      # (32, 32, 128)
        x = self.conv2dtrans4(x)      # (32, 32, 64)
        x = self.conv2dtrans5(x)      # (64, 64, 32)
        out = self.conv2dtrans_out(x) # (64, 64, 3)
        return out


class VariationalAutoEncoder(Model):

    def __init__(self, alpha=1.0, name='autoencoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder_VAE(alpha=alpha)
        self.decoder = Decoder(alpha=alpha)

    #def log_normal_pdf(self, sample, mean, logvar, raxis=1):
    #    log2pi = tf.math.log(2.0 * np.pi)
    #    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add Reconstruction loss.
        re_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        self.add_loss(re_loss)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        #logpz   = self.log_normal_pdf(z, 0.0, 0.0)
        #logqz_x = self.log_normal_pdf(z, z_mean, z_log_var)
        #kl_loss = logpz - logqz_x
        self.add_loss(kl_loss)
        return reconstructed


class RegularAutoEncoder(Model):

    def __init__(self, alpha=1.0, name='autoencoder', **kwargs):
        super(RegularAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(alpha=alpha)
        self.decoder = Decoder(alpha=alpha)

    def call(self, inputs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        re_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        self.add_loss(re_loss)
        return reconstructed


class Segmenter(Model):

    # TODO Suggested improvements:
    # > Use Transposed Convolution instead of UpSampling2D.
    def __init__(self, output_cnls=3, alpha=1.0, name='segmenter', **kwargs):
        super(Segmenter, self).__init__(name=name, **kwargs)
        self.conv0   = Conv2D(int(8*alpha), (1,1), strides=1, padding='same', activation=LeakyReLU(alpha=0.2), name='conv0')
        self.dc0     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc0')
        self.conv1   = Conv2D(int(16*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv1') # CRF = 3x3
        self.dc1     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc1')
        self.conv2   = Conv2D(int(16*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv2') # CRF = 5x5
        self.dc2     = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='dc2')
        self.conv3   = Conv2D(int(32*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv3') # CRF = 7x7
        self.dc3     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc3')
        self.conv4   = Conv2D(int(32*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv4') # CRF = 11x11
        self.dc4     = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='dc4')
        self.conv5   = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv5') # CRF = 15x15
        self.dc5     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc5')
        self.conv6   = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv6') # CRF = 23x23
        self.dc6     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc6')
        self.conv7   = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv7') # CRF = 31x31
        self.dc7     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc7')
        self.conv8   = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv8') # CRF = 39x39
        self.dc8     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc8')
        self.conv9   = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv9') # CRF = 47x47
        self.dc9     = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc9')
        self.conv10  = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv10') # CRF = 55x55
        self.dc10    = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc10')
        self.conv11  = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv11') # CRF = 63x63
        self.dc11    = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc11')
        self.conv12  = Conv2D(int(64*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv12') # CRF = 71x71
        #self.upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear') # upsample to 32x32x64
        self.upsamp0 = Conv2DTranspose(int(64*alpha), (3,3), strides=(2,2), padding='same', activation=LeakyReLU(alpha=0.2))
        self.dc12    = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc12')
        self.conv13  = Conv2D(int(32*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv13')
        #self.upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear') # upsample to 64x64x32
        self.upsamp1 = Conv2DTranspose(int(32*alpha), (3,3), strides=(2,2), padding='same', activation=LeakyReLU(alpha=0.2))
        self.dc13    = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc13')
        self.conv14  = Conv2D(int(16*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv14')
        self.cat7    = Concatenate(axis=-1) # concatenate skip connections
        self.dc14    = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name='dc14')
        self.conv15  = Conv2D(int(16*alpha), (1,1), padding='same', activation=LeakyReLU(alpha=0.2), name='conv15')
        self.conv16  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='conv16')

    def call(self, inputs):
        layer_seq = [self.conv0, self.dc0, self.conv1, self.dc1, self.conv2]
        x = inputs
        for layer in layer_seq:
            x = layer(x)
        conv2 = x
        layer_seq = [self.dc2, self.conv3, self.dc3, self.conv4, self.dc4, self.conv5, self.dc5, self.conv6, self.dc6, 
                     self.conv7, self.dc7, self.conv8, self.dc8, self.conv9, self.dc9, self.conv10, self.dc10, self.conv11, 
                     self.dc11, self.conv12, self.upsamp0, self.dc12, self.conv13, self.upsamp1, self.dc13, self.conv14]
        x = inputs
        for layer in layer_seq:
            x = layer(x)
        conv14 = x
        x = self.cat7([conv2, conv14])
        x = self.dc14(x)
        x = self.conv15(x)
        output = self.conv16(x)
        return output, conv2, conv14


class Composite(Model):

    def __init__(self, 
                 output_cnls=3, 
                 alpha=1.0, 
                 name='composite', 
                 encode_segmenter_output = True,
                 encode_segmenter_skips = False,
                 **kwargs):
        super(Composite, self).__init__(name=name, **kwargs)
        self.encode_segmenter_output = encode_segmenter_output
        self.encode_segmenter_skips  = encode_segmenter_skips
        self.segmenter = Segmenter(alpha = alpha, output_cnls = output_cnls, name = 'segmenter')
        self.ae        = RegularAutoEncoder(alpha = alpha, name = 'autoencoder')
        self.encoder   = self.ae.encoder
        self.decoder   = self.ae.decoder
        # -----------------------------------
        self.segmenter.trainable = False

    def call(self, inputs):
        x, seg_g = inputs # TODO Not sure it works this way. TODO 
        (seg_x_raw, features_x, deep_features_x) = self.segmenter(x)
        seg_x = seg_x_raw[:,:,:,0:2] # Remove the "fake-channel"
        if self.encode_segmenter_output and self.encode_segmenter_skips:
            ae_input = tf.concat([x, features_x, deep_features_x, seg_x], axis=-1)
        elif self.encode_segmenter_output:
            ae_input = tf.concat([x, seg_x], axis=-1)
        elif self.encode_segmenter_skips:
            ae_input = tf.concat([x, features_x, deep_features_x], axis=-1)
        else:
            ae_input = x
        z = self.encoder(ae_input)
        x_fake = self.decoder(z)
        (seg_x_raw_fake, features_x_fake, deep_features_x_fake) = self.segmenter(x_fake)
        seg_x_fake = seg_x_raw_fake[:,:,:,0:2] # Remove the "fake-channel"
        fake_cnl   = seg_x_raw_fake[:,:,:,2:3] # Separate out fake-channel.
        seg_g_wout_fake_cnl = seg_g[:,:,:,0:2] # Remove "fake" channel.
        seg_g_person        = seg_g[:,:,:,0:1]
        seg_g_car           = seg_g[:,:,:,1:2]
        seg_x_person        = seg_x[:,:,:,0:1]
        seg_x_car           = seg_x[:,:,:,1:2]
        seg_x_fake_person   = seg_x_fake[:,:,:,0:1]
        seg_x_fake_car      = seg_x_fake[:,:,:,1:2]
        loss_re             = tf.keras.losses.MeanSquaredError()(x, x_fake) # Reconstruction error.
        #loss_ammre_person  = Lambda(mabse_amm_img)([x, x_fake, seg_x_person])
        #loss_ammre_car     = Lambda(mabse_amm_img)([x, x_fake, seg_x_car])
        loss_output_conserv_person = tf.keras.losses.MeanSquaredError()(seg_x_person, seg_x_fake_person) # Output conservation error person.
        loss_output_conserv_car    = tf.keras.losses.MeanSquaredError()(seg_x_car, seg_x_fake_car) # Output conservation error car.
        #loss_feature_conserv       = tf.keras.losses.MeanSquaredError()([features_x, features_x_fake]) # Feature conservation error.
        #loss_deep_feature_conserv  = tf.keras.losses.MeanSquaredError()([deep_features_x, deep_features_x_fake]) # Deep feature conservation error.
        loss_fakeness              = K.mean((fake_cnl*fake_cnl), axis=[1,2,3], keepdims=False) # Error for producing a fake-looking image.
        # ^^^ Also note that we are not comparing apples-to-apples so, output_conserv losses can't be used with feature_conserv losses. TODO
        # ^^^ We can't, so instead we should just leave out feature losses for this paper. There's a way to normalize feature losses that involves
        #     training activation autoencoders that just map the activations directly back to themselves, but use tanh in their latent spaces.
        #     This opens up a lot of possibilities.
        iou_seg_x_seg_x_fake = K.sum(K.minimum(seg_x, seg_x_fake), axis=[1,2,3]) / K.sum(K.maximum(seg_x, seg_x_fake), axis=[1,2,3])
        iou_seg_g_seg_x      = K.sum(K.minimum(seg_g_wout_fake_cnl, seg_x), axis=[1,2,3]) / K.sum(K.maximum(seg_g_wout_fake_cnl, seg_x), axis=[1,2,3])
        iou_seg_g_seg_x_fake = K.sum(K.minimum(seg_g_wout_fake_cnl, seg_x_fake), axis=[1,2,3]) / K.sum(K.maximum(seg_g_wout_fake_cnl, seg_x_fake), axis=[1,2,3])
        self.add_loss(loss_re)
        self.add_loss(loss_output_conserv_person)
        self.add_loss(loss_output_conserv_car)
        self.add_loss(loss_fakeness)
        self.add_loss(iou_seg_x_seg_x_fake)
        self.add_loss(iou_seg_g_seg_x)
        self.add_loss(iou_seg_g_seg_x_fake)
        return x_fake, seg_x_raw, seg_x_raw_fake


def sampling(args): # TODO Need to attribute this properly. TODO
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_encoder_model(alpha = 1.0, in_cnls = 3):
    inputs = Input(shape=(64, 64, in_cnls))
    x = Conv2D(filters=int(32*alpha), kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=int(32*alpha), kernel_size=(5,5), activation=None, strides=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    layers = [(int(32*alpha)*2, (2,2)),
              (int(32*alpha)*2, (1,1)), 
              (int(32*alpha)*4, (2,2)), 
              (int(32*alpha)*4, (1,1)), 
              (int(32*alpha)*4, (1,1))]
    for n_filters, strides in layers:                                                                                       
        x = Conv2D(filters     = n_filters,
                   kernel_size = (5,5), 
                   activation  = None, 
                   strides     = strides, 
                   padding     = 'same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    z_mean    = Dense(48, name='z_mean_encode', activation=None)(x)
    z_log_var = Dense(48, name='z_log_var_encode', activation=None)(x)
    z         = Lambda(sampling, output_shape=(48,), name='z_encode')([z_mean, z_log_var])
    encoder = Model([inputs], [z, z_mean, z_log_var], name='encoder')
    encoder.summary()
    return encoder


def build_decoder_model(alpha = 1, output_cnls = 3):
    z_input = Input(shape=(48,))
    x = Dense(16*16*int(32*alpha)*4, activation=None)(z_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((16, 16, int(32*alpha)*4))(x)
    layers = [(int(32*alpha)*4, (1,1)),
              (int(32*alpha)*4, (1,1)),
              (int(32*alpha)*4, (2,2)),
              (int(32*alpha)*2, (1,1)), 
              (int(32*alpha)*2, (2,2)), 
              (int(32*alpha),   (1,1)), 
              (int(32*alpha),   (1,1))]
    for n_filters, strides in layers:
        x = Conv2DTranspose(filters=n_filters,
                            kernel_size=(5,5),
                            activation=None,
                            strides=strides,
                            padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    output = Conv2DTranspose(filters     = output_cnls, 
                             kernel_size = (5,5), 
                             activation  = 'tanh', 
                             strides     = (1,1), 
                             padding     = 'same')(x)
    decoder = Model(z_input, output)
    decoder.summary()
    return decoder


def build_encoder_model_v2(alpha = 1, in_cnls = 3):
    inputs = Input(shape=(64,64,3))
    x = Conv2D(int(8*alpha), (1,1), strides=1, padding='same', activation='relu')(inputs)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(16*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 3x3
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(16*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 5x5
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same')(x) # (32, 32, 16)
    x = Conv2D(int(32*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 7x7
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(32*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 11x11
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same')(x) # (16, 16, 32)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 15x15
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 23x23
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 31x31
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 39x39
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 47x47
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 55x55
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 63x63
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # CRF = 71x71
    z_mean = DepthwiseConv2D(kernel_size=5, strides=4, activation=None, padding='same')(x) # (4, 4, 64)
    z_mean = Conv2D(3, (1,1), padding='same', activation=None, name='z_mean')(z_mean) # (4, 4, 3)
    z_log_var = DepthwiseConv2D(kernel_size=5, strides=4, activation=None, padding='same')(x)
    z_log_var = Conv2D(3, (1,1), padding='same', activation=None, name='z_log_var')(z_log_var)
    #z_mean = K.cast(z_mean, 'float32')
    #z_log_var = K.cast(z_log_var, 'float32')
    z_mean_flat    = Flatten()(z_mean) # (48, )
    z_log_var_flat = Flatten()(z_log_var) # (48, )
    #z_mean_flat    = Reshape((48,))(z_mean) # (48, )
    #z_log_var_flat = Reshape((48,))(z_log_var) # (48, )
    z = Lambda(sampling, output_shape=(48,), name='z_encode')([z_mean_flat, z_log_var_flat])
    encoder = Model([inputs], [z, z_mean_flat, z_log_var_flat], name='encoder')
    encoder.summary()
    return encoder


def build_decoder_model_v2(alpha       = 1,
                           output_cnls = 3,
                           out_act     = 'tanh'):
    z_input = Input(shape=(48,))
    x = Reshape((4, 4, 3))(z_input)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x) # (4, 4, 64)
    x = UpSampling2D(size=(4,4), interpolation='bilinear')(x) # (16, 16, 64)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(64*alpha), (1,1), padding='same', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x) # (16, 16, 64)
    x = Conv2D(int(32*alpha), (1,1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x) # upsample to (32, 32, 32)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(32*alpha), (1,1), padding='same', activation='relu')(x) # (32, 32, 32)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(16*alpha), (1,1), padding='same', activation='relu')(x) # (32, 32, 16)
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x) # upsample to (64, 64, 16)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(16*alpha), (1,1), padding='same', activation='relu')(x) # (64, 64, 16)
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same')(x)
    x = Conv2D(int(16*alpha), (1,1), padding='same', activation='relu')(x) # (64, 64, 16)
    output = Conv2D(output_cnls, (1,1), padding='same', activation='tanh')(x) # (64, 64, 3)
    decoder = Model(z_input, output)
    decoder.summary()
    return decoder


def build_vae_model(alpha        = 1, 
                    latent_dims  = 48,
                    input_cnls   = 3,
                    output_cnls  = 3):
    encoder = build_encoder_model_v2(alpha        = alpha, 
                                     latent_dims  = latent_dims, 
                                     in_cnls      = input_cnls)
    decoder = build_decoder_model_v2(alpha        = alpha,
                                     latent_dims  = latent_dims,
                                     output_cnls  = output_cnls)
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
    #segmenter_training, segmenter = build_segmenter_model(filters = filters, output_cnls = 3)
    segmenter_training, segmenter = build_segmenter_model_mobilenetv1_64x64(alpha = 1, output_cnls = 3)
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
        kl_loss = 1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_)
        #kl_loss = K.sum(kl_loss, axis=-1)
        #kl_loss *= -0.5
        #kl_loss = K.mean(kl_loss)
        kl_loss = K.sum(kl_loss, axis=[1], keepdims=False)
        kl_loss *= -0.5
        kl_loss = K.mean(kl_loss, keepdims=True)
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
        return K.mean(K.square(x_ - x_fake_), axis=[1,2,3], keepdims=False)

    def mabse_img(args):
        """ Mean absolute error. Used for unbounded functions. """
        x_, x_fake_ = args
        return K.mean(K.abs(x_ - x_fake_), axis=[1,2,3], keepdims=False)

    def mse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return K.mean(K.square((x_ - x_fake_) * attn_map), axis=[1,2,3], keepdims=False)

    def mabse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return K.mean(K.abs((x_ - x_fake_) * attn_map), axis=[1,2,3], keepdims=False)

    def intersection_over_union(args): # TODO Is there a better way to do axis=[1,2,3]?
        y_true, y_pred = args
        return K.sum(K.minimum(y_true, y_pred), axis=[1,2,3]) / K.sum(K.maximum(y_true, y_pred), axis=[1,2,3])

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
    loss_fakeness              = Lambda(lambda t: K.mean(t, axis=[1,2,3], keepdims=False))(fake_cnl)
    # ^^^ should these be mean-square-error or mean-absolute-error, since they're unbounded functions. TODO
    # ^^^ Also note that we are not comparing apples-to-apples so, output_conserv losses can't be used with feature_conserv losses. TODO
    loss_kl                    = Lambda(kl_divergence)([z1, z_mean1, z_log_var1]) # KL-divergence loss.
    iou_seg_x_seg_x_fake       = Lambda(intersection_over_union)([seg_x, seg_x_fake])
    iou_seg_g_seg_x            = Lambda(intersection_over_union)([seg_g_wout_fake_cnl, seg_x])
    iou_seg_g_seg_x_fake       = Lambda(intersection_over_union)([seg_g_wout_fake_cnl, seg_x_fake])
    loss_composite             = K.mean(loss_output_conserv_person * w_loss_output_conserv_person \
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





# ==========================================================================================

def build_segmenter_model_aspp(alpha=1, output_cnls=3):
    inputs  = Input(shape=(128,128,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc2d0_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d0_1')(conv0)
    dc2d0_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d0_2')(conv0)
    dc2d0_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d0_3')(conv0)
    dc2d0_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d0_4')(conv0)
    dc2d0_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d0_5')(conv0)
    cat0    = Concatenate(axis=-1)([dc2d0_1, dc2d0_2, dc2d0_3, dc2d0_4, dc2d0_5])
    aspp0   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp0')(cat0) # CRF = 11x11
    dc2d1_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d1_1')(aspp0)
    dc2d1_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d1_2')(aspp0)
    dc2d1_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d1_3')(aspp0)
    dc2d1_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d1_4')(aspp0)
    dc2d1_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d1_5')(aspp0)
    cat1    = Concatenate(axis=-1)([dc2d1_1, dc2d1_2, dc2d1_3, dc2d1_4, dc2d1_5])
    aspp1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp1')(cat1) # CRF = 21x21
    # vvv Downsample to 64 x 64
    #dc2d2_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d2_1')(aspp1)
    #dc2d2_2 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d2_2')(aspp1)
    #dc2d2_3 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d2_3')(aspp1) # (65, 65, 32)
    #dc2d2_4 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d2_4')(aspp1)
    #dc2d2_5 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d2_5')(aspp1) # (63, 63, 32)
    #cat2    = Concatenate(axis=-1)([dc2d2_1, dc2d2_2, dc2d2_3, dc2d2_4, dc2d2_5])
    cat2    = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='cat2')(aspp1)
    aspp2   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp2')(cat2) # CRF = 22x22
    dc2d3_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d3_1')(aspp2)
    dc2d3_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d3_2')(aspp2)
    dc2d3_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d3_3')(aspp2)
    dc2d3_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d3_4')(aspp2)
    dc2d3_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d3_5')(aspp2)
    cat3    = Concatenate(axis=-1)([dc2d3_1, dc2d3_2, dc2d3_3, dc2d3_4, dc2d3_5])
    aspp3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp3')(cat3) # CRF = 42x42
    # vvv Downsample to 32 x 32
    #dc2d4_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d4_1')(aspp3)
    #dc2d4_2 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d4_2')(aspp3)
    #dc2d4_3 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d4_3')(aspp3)
    #dc2d4_4 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d4_4')(aspp3)
    #dc2d4_5 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d4_5')(aspp3)
    #cat4    = Concatenate(axis=-1)([dc2d4_1, dc2d4_2, dc2d4_3, dc2d4_4, dc2d4_5])
    cat4    = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='cat4')(aspp3)
    aspp4   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp4')(cat4) # CRF = 43x43
    dc2d5_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d5_1')(aspp4)
    dc2d5_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d5_2')(aspp4)
    dc2d5_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d5_3')(aspp4)
    dc2d5_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d5_4')(aspp4)
    dc2d5_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d5_5')(aspp4)
    cat5    = Concatenate(axis=-1)([dc2d5_1, dc2d5_2, dc2d5_3, dc2d5_4, dc2d5_5])
    aspp5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp5')(cat5) # CRF = 83x83
    dc2d6_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d6_1')(aspp5)
    dc2d6_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d6_2')(aspp5)
    dc2d6_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d6_3')(aspp5)
    dc2d6_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d6_4')(aspp5)
    dc2d6_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d6_5')(aspp5)
    cat6    = Concatenate(axis=-1)([dc2d6_1, dc2d6_2, dc2d6_3, dc2d6_4, dc2d6_5])
    aspp6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp6')(cat6) # CRF = 123x123
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(aspp6) # upsample to 64x64x256
    # optional concat, skipped for now, TODO
    dsc0    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc0_')(upsamp0)
    dsc0    = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='dsc0')(dsc0)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(dsc0) # upsample to 128x128x128
    dsc1    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc1_')(upsamp1)
    dsc1    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc1')(dsc1)
    cat7    = Concatenate(axis=-1)([dsc1, aspp1]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    #dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='sigmoid', name='output')(dsc2)
    model_skip     = Model([inputs], [output, aspp1, dsc1])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_aspp_64x64(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc2d0_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d0_1')(conv0)
    dc2d0_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d0_2')(conv0)
    dc2d0_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d0_3')(conv0)
    dc2d0_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d0_4')(conv0)
    dc2d0_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d0_5')(conv0)
    cat0    = Concatenate(axis=-1)([dc2d0_1, dc2d0_2, dc2d0_3, dc2d0_4, dc2d0_5])
    aspp0   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp0')(cat0) # CRF = 11x11
    dc2d1_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d1_1')(aspp0)
    dc2d1_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d1_2')(aspp0)
    dc2d1_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d1_3')(aspp0)
    dc2d1_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d1_4')(aspp0)
    dc2d1_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d1_5')(aspp0)
    cat1    = Concatenate(axis=-1)([dc2d1_1, dc2d1_2, dc2d1_3, dc2d1_4, dc2d1_5])
    aspp1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp1')(cat1) # CRF = 21x21
    # vvv Downsample to 32 x 32
    #dc2d2_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d2_1')(aspp1)
    #dc2d2_2 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d2_2')(aspp1)
    #dc2d2_3 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d2_3')(aspp1) # (65, 65, 32)
    #dc2d2_4 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d2_4')(aspp1)
    #dc2d2_5 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d2_5')(aspp1) # (63, 63, 32)
    #cat2    = Concatenate(axis=-1)([dc2d2_1, dc2d2_2, dc2d2_3, dc2d2_4, dc2d2_5])
    cat2    = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='cat2')(aspp1)
    aspp2   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp2')(cat2) # CRF = 22x22
    dc2d3_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d3_1')(aspp2)
    dc2d3_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d3_2')(aspp2)
    dc2d3_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d3_3')(aspp2)
    dc2d3_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d3_4')(aspp2)
    dc2d3_5 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d3_5')(aspp2)
    cat3    = Concatenate(axis=-1)([dc2d3_1, dc2d3_2, dc2d3_3, dc2d3_4, dc2d3_5])
    aspp3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp3')(cat3) # CRF = 42x42
    # vvv Downsample to 16 x 16
    #dc2d4_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d4_1')(aspp3)
    #dc2d4_2 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d4_2')(aspp3)
    #dc2d4_3 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d4_3')(aspp3)
    #dc2d4_4 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d4_4')(aspp3)
    #dc2d4_5 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(5, 5), name='dc2d4_5')(aspp3)
    #cat4    = Concatenate(axis=-1)([dc2d4_1, dc2d4_2, dc2d4_3, dc2d4_4, dc2d4_5])
    cat4    = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='cat4')(aspp3)
    aspp4   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp4')(cat4) # CRF = 43x43
    dc2d5_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d5_1')(aspp4)
    dc2d5_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d5_2')(aspp4)
    dc2d5_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d5_3')(aspp4)
    dc2d5_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d5_4')(aspp4)
    cat5    = Concatenate(axis=-1)([dc2d5_1, dc2d5_2, dc2d5_3, dc2d5_4])
    aspp5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp5')(cat5) # CRF = 83x83
    dc2d6_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d6_1')(aspp5)
    dc2d6_2 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d6_2')(aspp5)
    dc2d6_3 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(3, 3), name='dc2d6_3')(aspp5)
    dc2d6_4 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(4, 4), name='dc2d6_4')(aspp5)
    cat6    = Concatenate(axis=-1)([dc2d6_1, dc2d6_2, dc2d6_3, dc2d6_4])
    aspp6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp6')(cat6) # CRF = 123x123
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(aspp6) # upsample to 32x32x64
    # optional concat, skipped for now, TODO
    dsc0    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc0_')(upsamp0)
    dsc0    = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='dsc0')(dsc0)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(dsc0) # upsample to 64x64x32
    dsc1    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc1_')(upsamp1)
    dsc1    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc1')(dsc1)
    cat7    = Concatenate(axis=-1)([dsc1, aspp1]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    #dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dsc2)
    model_skip     = Model([inputs], [output, aspp1, dsc1])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_mobilenetv1_64x64(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc0     = DepthwiseConv2D(kernel_size=11, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc0')(conv0)
    conv1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv1')(dc0) # CRF = 11x11
    dc1     = DepthwiseConv2D(kernel_size=11, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc1')(conv1)
    conv2   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv2')(dc1) # CRF = 21x21
    dc2     = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2')(conv2)
    conv3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv3')(dc2) # CRF = 22x22
    dc3     = DepthwiseConv2D(kernel_size=11, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc3')(conv3)
    conv4   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv4')(dc3) # CRF = 42x42
    dc4     = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc4')(conv4)
    conv5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv5')(dc4) # CRF = 43x43
    dc5     = DepthwiseConv2D(kernel_size=11, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc5')(conv5)
    conv6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv6')(dc5) # CRF = 83x83
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv6) # upsample to 32x32x64
    dc6     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc6')(upsamp0)
    conv7   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv7')(dc6)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv7) # upsample to 64x64x32
    dc7     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc7')(upsamp1)
    conv8   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv8')(dc7)
    cat7    = Concatenate(axis=-1)([conv2, conv8]) # concatenate skip connections
    dc8     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc8')(cat7)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dc8)
    model_skip     = Model([inputs], [output, conv2, conv6])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_mobilenetv1_64x64_v2(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc0     = DepthwiseConv2D(kernel_size=9, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc0')(conv0)
    conv1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv1')(dc0) # CRF = 11x11
    dc1     = DepthwiseConv2D(kernel_size=9, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc1')(conv1)
    conv2   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv2')(dc1) # CRF = 21x21
    dc2     = DepthwiseConv2D(kernel_size=9, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2')(conv2)
    conv3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv3')(dc2) # CRF = 22x22
    dc3     = DepthwiseConv2D(kernel_size=9, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc3')(conv3)
    conv4   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv4')(dc3) # CRF = 42x42
    dc4     = DepthwiseConv2D(kernel_size=9, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc4')(conv4)
    conv5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv5')(dc4) # CRF = 43x43
    dc5     = DepthwiseConv2D(kernel_size=9, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc5')(conv5)
    conv6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv6')(dc5) # CRF = 83x83
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv6) # upsample to 32x32x64
    dc6     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc6')(upsamp0)
    conv7   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv7')(dc6)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv7) # upsample to 64x64x32
    dc7     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc7')(upsamp1)
    conv8   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv8')(dc7)
    cat7    = Concatenate(axis=-1)([conv2, conv8]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dsc2)
    model_skip     = Model([inputs], [output, conv2, conv6])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_mobilenetv1_64x64_v3(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc0     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc0')(conv0)
    conv1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv1')(dc0) # CRF = 5x5
    dc1     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc1')(conv1)
    conv2   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv2')(dc1) # CRF = 9x9
    dc2     = DepthwiseConv2D(kernel_size=5, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2')(conv2)
    conv3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv3')(dc2) # CRF = 13x13
    dc3     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc3')(conv3)
    conv4   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv4')(dc3) # CRF = 21x21
    dc4     = DepthwiseConv2D(kernel_size=5, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc4')(conv4)
    conv5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv5')(dc4) # CRF = 29x29
    dc5     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc5')(conv5)
    conv6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv6')(dc5) # CRF = 45x45
    dc6     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc6')(conv6)
    conv7   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv7')(dc6) # CRF = 61x61
    dc7     = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc7')(conv7)
    conv8   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv8')(dc7) # CRF = 77x77
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv8) # upsample to 32x32x64
    dc6     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc8')(upsamp0)
    conv9   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv9')(dc6)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv9) # upsample to 64x64x32
    dc7     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc9')(upsamp1)
    conv10  = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv10')(dc7)
    cat7    = Concatenate(axis=-1)([conv2, conv10]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dsc2)
    model_skip     = Model([inputs], [output, conv2, conv8])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_mobilenetv1_64x64_v4(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc0     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc0')(conv0)
    conv1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv1')(dc0) # CRF = 3x3
    dc1     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc1')(conv1)
    conv2   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv2')(dc1) # CRF = 5x5
    dc2     = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2')(conv2)
    conv3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv3')(dc2) # CRF = 7x7
    dc3     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc3')(conv3)
    conv4   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv4')(dc3) # CRF = 11x11
    dc4     = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc4')(conv4)
    conv5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv5')(dc4) # CRF = 15x15
    dc5     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc5')(conv5)
    conv6   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv6')(dc5) # CRF = 23x23
    dc6     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc6')(conv6)
    conv7   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv7')(dc6) # CRF = 31x31
    dc7     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc7')(conv7)
    conv8   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv8')(dc7) # CRF = 39x39
    dc8     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc8')(conv8)
    conv9   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv9')(dc8) # CRF = 47x47
    dc9     = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc9')(conv9)
    conv10  = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv10')(dc9) # CRF = 55x55
    dc10    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc10')(conv10)
    conv11  = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv11')(dc10) # CRF = 63x63
    dc11    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc11')(conv11)
    conv12  = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='conv12')(dc11) # CRF = 71x71
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv12) # upsample to 32x32x64
    dc12    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc12')(upsamp0)
    conv13  = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='conv13')(dc12)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(conv13) # upsample to 64x64x32
    dc13    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc13')(upsamp1)
    conv14  = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='conv14')(dc13)
    cat7    = Concatenate(axis=-1)([conv2, conv14]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dsc2)
    model_skip     = Model([inputs], [output, conv2, conv12])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip


def build_segmenter_model_aspp_64x64_v2(alpha=1, output_cnls=3):
    inputs  = Input(shape=(64,64,3))
    conv0   = Conv2D(8*alpha, (1,1), strides=1, padding='same', activation='relu', name='conv0')(inputs)
    dc2d0_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d0_1')(conv0)
    dc2d0_2 = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d0_2')(conv0)
    cat0    = Concatenate(axis=-1)([dc2d0_1, dc2d0_2])
    aspp0   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp0')(cat0) # CRF = 9x9
    dc2d1_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d1_1')(aspp0)
    dc2d1_2 = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d1_2')(aspp0)
    cat1    = Concatenate(axis=-1)([dc2d1_1, dc2d1_2])
    aspp1   = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='aspp1')(cat1) # CRF = 17x17
    # vvv Downsample to 32 x 32
    dc2d2_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d2_1')(aspp1)
    dc2d2_2 = DepthwiseConv2D(kernel_size=5, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d2_2')(aspp1)
    cat2    = Concatenate(axis=-1)([dc2d2_1, dc2d2_2])
    aspp2   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp2')(cat2) # CRF = 25x25
    dc2d3_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d3_1')(aspp2)
    dc2d3_2 = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d3_2')(aspp2)
    cat3    = Concatenate(axis=-1)([dc2d3_1, dc2d3_2])
    aspp3   = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='aspp3')(cat3) # CRF = 41x41
    # vvv Downsample to 16 x 16
    dc2d4_1 = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d4_1')(aspp3)
    dc2d4_2 = DepthwiseConv2D(kernel_size=5, strides=2, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d4_2')(aspp3)
    cat4    = Concatenate(axis=-1)([dc2d4_1, dc2d4_2])
    aspp4   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp4')(cat4) # CRF = 57x57
    dc2d5_1 = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dc2d5_1')(aspp4)
    dc2d5_2 = DepthwiseConv2D(kernel_size=5, strides=1, activation=None, padding='same', dilation_rate=(2, 2), name='dc2d5_2')(aspp4)
    cat5    = Concatenate(axis=-1)([dc2d5_1, dc2d5_2])
    aspp5   = Conv2D(64*alpha, (1,1), padding='same', activation='relu', name='aspp5')(cat5) # CRF = 89x89
    upsamp0 = UpSampling2D(size=(2,2), interpolation='bilinear')(aspp5) # upsample to 32x32x64
    # optional concat, skipped for now, TODO
    dsc0    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc0_')(upsamp0)
    dsc0    = Conv2D(32*alpha, (1,1), padding='same', activation='relu', name='dsc0')(dsc0)
    upsamp1 = UpSampling2D(size=(2,2), interpolation='bilinear')(dsc0) # upsample to 64x64x32
    dsc1    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc1_')(upsamp1)
    dsc1    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc1')(dsc1)
    cat7    = Concatenate(axis=-1)([dsc1, aspp1]) # concatenate skip connections
    dsc2    = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding='same', dilation_rate=(1, 1), name='dsc2_')(cat7)
    dsc2    = Conv2D(16*alpha, (1,1), padding='same', activation='relu', name='dsc2')(dsc2)
    output  = Conv2D(output_cnls, (1,1), padding='same', activation='tanh', name='output')(dsc2)
    model_skip     = Model([inputs], [output, aspp1, aspp5])
    model_training = Model([inputs], [output])
    model_skip.summary()
    return model_training, model_skip
    # ^^^ I have a feeling the bilinear upsampling is making it blurry. Consider switching to convolution...




    
