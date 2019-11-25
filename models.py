import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, Subtract, Lambda, ReLU
from keras.layers import Reshape, Conv2DTranspose, Activation, Concatenate
from keras.layers.advanced_activations import LeakyReLU, ThresholdedReLU
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam


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
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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
    z         = Lambda(sampling, output_shape=(latent_dims,), name='z_encode')([z_mean, z_log_var, input_scale_var])
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
                    input_cnls   = 3
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
                          w_loss_kl_divergence         = 0.5,
                          filters                      = 32,
                          encode_segmenter_output      = True,
                          encode_segmenter_skips       = False):
    # ---- Define and load segmenter model ----
    segmenter_training, segmenter = build_segmenter_model(filters = filters, output_cnls = 2)
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
    z, z_mean_, z_log_var_ = encoder(input_img)
    decoder_output = decoder(z)
    vae = Model([inputs_vae], [decoder_output, z_mean_, z_log_var_], name='vae')
    if path_pretrained_encoder:
        encoder.load_weights(path_pretrained_encoder)
    if path_pretrained_decoder:
        decoder.load_weights(path_pretrained_decoder)
    # ---- Define composite model ----
    x = Input(shape=(64, 64, 3))
    (seg_x, features_x, deep_features_x) = segmenter(x)
    seg_x = Lambda(lambda t: (t + 1.0) / 2.0)(seg_x) # Remap from -1.0 -> 1.0 to 0.0 -> 1.0
    if encode_segmenter_output and encode_segmenter_skips:
        vae_input = Concatenate(axis=-1)([x, features_x, deep_features_x, seg_x])
    elif encode_segmenter_output:
        vae_input = Concatenate(axis=-1)([x, seg_x])
    elif encode_segmenter_skips:
        vae_input = Concatenate(axis=-1)([x, features_x, deep_features_x])
    else:
        vae_input = x
    x_fake, z_mean, z_log_var = vae(vae_input)
    (seg_x_fake, features_x_fake, deep_features_x_fake) = segmenter(x_fake)
    seg_x_fake = Lambda(lambda t: (t + 1.0) / 2.0)(seg_x_fake) # Remap from -1.0 -> 1.0 to 0.0 -> 1.0

    # http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    def kl_divergence(args):
        z_mean_, z_log_var_ = args
        kl_loss = 1 + z_log_var_ - K.square(z_mean_) - K.exp(z_log_var_)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = K.mean(kl_loss)
        return kl_loss

    def mse_img(args):
        """ Mean square error. Used for bounded functions. """
        x_, x_fake_ = args
        return K.mean(K.square(x_ - x_fake_))

    def mabse_img(args):
        """ Mean absolute error. Used for unbounded functions. """
        x_, x_fake_ = args
        return K.mean(K.abs(x_ - x_fake_))

    def mse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return K.mean(K.square((x_ - x_fake_) * attn_map))

    def mabse_amm_img(args):
        """ Attention-map-modulated mean square error 

        # Note
            attn_map needs to be of shape (64, 64, 1)
        """
        x_, x_fake_, attn_map = args
        return K.mean(K.abs((x_ - x_fake_) * attn_map))

    def intersection_over_union(args): # TODO Is there a better way to do axis=[1,2,3]?
        y_true, y_pred = args
        return K.sum(K.minimum(y_true, y_pred), axis=[1,2,3]) / K.sum(K.maximum(y_true, y_pred), axis=[1,2,3])

    seg_g        = Input(shape=(64, 64, 2)) # Segmentation ground truth.
    seg_g_person = Lambda(lambda t: t[:,:,:,0:1])(seg_g)
    seg_g_car    = Lambda(lambda t: t[:,:,:,1:2])(seg_g)
    seg_x_person = Lambda(lambda t: t[:,:,:,0:1])(seg_x)
    seg_x_car    = Lambda(lambda t: t[:,:,:,1:2])(seg_x)
    seg_x_fake_person = Lambda(lambda t: t[:,:,:,0:1])(seg_x_fake)
    seg_x_fake_car    = Lambda(lambda t: t[:,:,:,1:2])(seg_x_fake)
    loss_re                    = Lambda(mabse_img)([x, x_fake]) # Reconstruction error.
    loss_ammre_person          = Lambda(mabse_amm_img)([x, x_fake, seg_x_person])
    loss_ammre_car             = Lambda(mabse_amm_img)([x, x_fake, seg_x_car])
    loss_output_conserv_person = Lambda(mse_img)([seg_x_person, seg_x_fake_person]) # Output conservation error person.
    loss_output_conserv_car    = Lambda(mse_img)([seg_x_car, seg_x_fake_car]) # Output conservation error car.
    loss_feature_conserv       = Lambda(mabse_img)([features_x, features_x_fake]) # Feature conservation error.
    loss_deep_feature_conserv  = Lambda(mabse_img)([deep_features_x, deep_features_x_fake]) # Deep feature conservation error.
    # ^^^ should these be mean-square-error or mean-absolute-error, since they're unbounded functions. TODO
    # ^^^ Also note that we are not comparing apples-to-apples so, output_conserv losses can't be used with feature_conserv losses. TODO
    loss_kl                    = Lambda(kl_divergence)([z_mean, z_log_var]) # KL-divergence loss.
    iou_seg_x_seg_x_fake       = Lambda(intersection_over_union)([seg_x, seg_x_fake])
    iou_seg_g_seg_x            = Lambda(intersection_over_union)([seg_x_g, seg_x])
    iou_seg_g_seg_x_fake       = Lambda(intersection_over_union)([seg_x_g, seg_x_fake])
    loss_composite             = K.mean(loss_output_conserv_person * w_loss_output_conserv_person \
                                      + loss_output_conserv_car    * w_loss_output_conserv_car \
                                      + loss_feature_conserv       * w_loss_feature_conserv \
                                      + loss_deep_feature_conserv  * w_loss_deep_feature_conserv \
                                      + loss_kl                    * w_loss_kl_divergence) # TODO Why mean?
    composite_outputs = [x_fake, 
                         seg_x, 
                         seg_x_fake, 
                         loss_re, 
                         loss_feature_conserv, 
                         loss_deep_feature_conserv, 
                         loss_output_conserv_person,
                         loss_output_conserv_car, 
                         loss_kl, 
                         iou_seg_x_seg_x_fake,
                         iou_seg_g_seg_x,
                         iou_seg_g_seg_x_fake]
    composite = Model([x, seg_g], composite_outputs)
    composite.add_loss(loss_composite)
    composite.summary()
    composite.compile(optimizer = Adam(0.0001, 0.5))
    return segmenter_training, vae, encoder, decoder, composite


