"""

"""

from __future__ import print_function, division

import sys, os

if len(sys.argv) < 2:
    print('please specify <gpu_number>')
    exit()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import keras
import numpy as np
import pickle
import cv2

from models import build_composite_model


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


def train(save_path,
          batch_size                   = 64,
          max_epochs                   = 100,
          w_loss_feature_conserv       = 0.25,
          w_loss_deep_feature_conserv  = 0.25,
          w_loss_output_conserv_person = 0.0,
          w_loss_output_conserv_car    = 0.0,
          w_loss_kl_divergence         = 0.5,
          encode_segmenter_output      = True,
          encode_segmenter_skips       = False,
          path_pretrained_segmenter    = None,
          path_pretrained_encoder      = None,
          path_pretrained_decoder      = None):
    # create save directories if they don't exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path+'/model_saves/')
        os.mkdir(save_path+'/images/')
    # load and normalize dataset
    with open('train_64000_64px.npz', 'rb') as f:
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
    # build composite model for training autoencoder
    args = {'path_pretrained_segmenter'   : path_pretrained_segmenter,
            'path_pretrained_encoder'     : path_pretrained_encoder,
            'path_pretrained_decoder'     : path_pretrained_decoder,
            'w_loss_feature_conserv'      : w_loss_feature_conserv,
            'w_loss_deep_feature_conserv' : w_loss_deep_feature_conserv,
            'w_loss_output_conserv_person': w_loss_output_conserv_person,
            'w_loss_output_conserv_car'   : w_loss_output_conserv_car,
            'w_loss_kl_divergence'        : w_loss_kl_divergence,
            'filters'                     : 32,
            'encode_segmenter_output'     : encode_segmenter_output,
            'encode_segmenter_skips'      : encode_segmenter_skips
           }
    #segmenter, vae, encoder, decoder, composite = build_composite_model(**args)
    segmenter, vae, encoder, decoder, composite = build_composite_model(**args)
    batches_per_epoch = x_train.shape[0] // batch_size
    seg_best, composite_best = 10000.0, 10000.0
    for idx_epoch in range(max_epochs):
        x_train, y_train = shuffle_in_unison(x_train, y_train)
        for idx_batch in range(x_train.shape[0] // batch_size):
            from_, to_ = idx_batch*batch_size, (idx_batch+1)*batch_size
            x_batch, y_batch = x_train[from_:to_], y_train[from_:to_]
            # IDEA: Segmenter has a "fake" channel.
            # How do we label fake samples? Car + person + fake? or just fake?
            # How strong is the penalty for fake?
            # Do we mix parts of images so half an image is real and half is fake or something?
            # Some images will be all real or all fake, and others will have a random rectangular region that's either real or fake.
            # Choose either a random horizontal line or a random vertical line within a certain range.
            # TODO TODO ^^^^
            ret = composite.predict([x_batch[0:16], y_batch[0:16]])
            x_fake, seg_x, seg_x_fake = ret[0:3] 
            loss_re, loss_feature_conserv, loss_deep_feature_conserv = ret[3:6] 
            loss_output_conserv_person, loss_output_conserv_car = ret[6:8]
            loss_kl, iou_seg_x_seg_x_fake, iou_seg_g_seg_x, iou_seg_g_seg_x_fake = ret[8:]
            # loss_re, loss_ammre, loss_attn = np.mean(loss_re), np.mean(loss_ammre), np.mean(loss_attn)
            # iou_mean_x, iou_mean_fake = np.mean(iou_attn_x), np.mean(iou_attn_fake)
            # Replace 2 images with fake versions
            pass
            # Use the other 14 to augment their images with a random blend. (i.e. horizontal or vertical line merge)
            pass
            # Update the hallucination channel of the y_batch to reflect the augmentation
            pass
            # train models on the batch
            seg_loss  = segmenter.train_on_batch(x=x_batch, y=y_batch)
            comp_loss = composite.train_on_batch(x=[x_batch, y_batch], y=None)
        # TODO print(idx_epoch, len(adv_imgs), loss_re, loss_ammre, loss_attn, iou_mean_x, iou_mean_fake)
        seg_acc, seg_loss = segmenter.evaluate(x=x_val, y=y_val) # TODO verify.
        # Evaluate the composite model somehow:
        pass
        if seg_loss < seg_best:
            attention.save_weights(save_path+'/model_saves/attention_weights_'+str(idx_batch)+'.h5')
        if comp_loss < composite_best: # TODO Doesn't seem like a good metric. TODO
            encoder.save_weights(save_path+'/model_saves/encoder_weights_'+str(idx_batch)+'.h5')
            decoder.save_weights(save_path+'/model_saves/decoder_weights_'+str(idx_batch)+'.h5')

        if idx_batch % test_interval == 0:
            test_exemplars = [11,12,16,18,20,23,24,44,45,60,66,69,102]
            test_exemplars.extend([103,106,107,109,112,116,132,139,150,160,163,164])
            test_exemplars.extend([169,176,177,178,180,188,203,233,238,243,260,267])
            test_exemplars.extend([269,295,296,338,339,340,367,385,422,426,531,568,599])
            print(len(test_exemplars))
            test_exemplars = np.array(test_exemplars, dtype=np.int)
            x_test_batch = x_test[test_exemplars]
            y_test_batch = y_test[test_exemplars]
            x_fake, attn_x_fake, corrupted_img, loss_re, loss_ammre, loss_attn, iou_attn_x, iou_attn_fake = composite.predict([x_test_batch, y_test_batch])
            iou_attn_x = str(round(np.mean(iou_attn_x), 3)).replace('.', '_')
            iou_attn_fake = str(round(np.mean(iou_attn_fake), 3)).replace('.', '_')
            attn_x = attention.predict(x_test_batch)
            attn_x = [(img * 2.0) - 1.0 for img in attn_x]
            attn_x_fake = [(img * 2.0) - 1.0 for img in attn_x_fake]
            save_images(save_path+'/images/'+str(idx_batch)+'_fake.jpg', x_test_batch, x_fake)
            save_images(save_path+'/images/'+str(idx_batch)+'_attention_'+iou_attn_x+'.jpg', x_test_batch, attn_x)
            save_images(save_path+'/images/'+str(idx_batch)+'_attention_fake_'+iou_attn_fake+'.jpg', x_test_batch, attn_x_fake)
            save_images(save_path+'/images/'+str(idx_batch)+'_corrupted.jpg', x_test_batch, corrupted_img)


# Experiment list:
# Next: Try hyperbolic tangent latent activations. (try it on the larger model after validating on smaller model)
#   ^^^ Doesn't help. Nothing helps except lowering the learning rate. Try a different optimizer.
#       Adadelta allows reliable convergence of the large autoencoder, but gives unstable convergence for the attention model.
#       Consider using adam(lr=0.0001, 0.5) for the attention model and adadelta for the autoencoder.
# Next: Try training alongside attention model versus pretrained attention.
#   ^^^ Results are strikingly better when training alongside the attention model. Might be because of a more
#       robust gradient. Might be because attention model isn't overfitting as much. The slow rate of attention
#       model training (batches of 16) may also be a big part of this.

# TODO Try different loss functions again.
# TODO Might it work better without the attention model being pretrained because more robust gradient? TODO
# TODO Do a more in-depth investigation of why batch-norm didn't work. TODO
# TODO Make experimental rig a lot more flexible and log everything properly. TODO

# TODO And update the git repo, merge the changes in from the new one.
if __name__ == '__main__':
    
    train(batch_size                  = 64,
          test_interval               = 250,
          adversarial_samples         = 'x_fake',
          max_adversarial             = 0,
          d_thermostat                = 0.01,
          w_loss_re                   = 0.0, 
          w_loss_ammre                = 0.5,
          w_loss_perceptual           = 0.0,
          w_loss_rea                  = 0.5,
          num_batches                 = 50001,
          save_interval               = 25000,
          path_pretrained_autoencoder = None,
          path_pretrained_attention   = None,
          path_pretrained_encoder     = 'models/pretrained_encoder_64px_64f_48z_run3.h5',
          path_pretrained_decoder     = 'models/pretrained_decoder_64px_64f_48z_run3.h5',
          save_path                   = 'self_adversarial_test_00_05_00_05'
         )
    # 'models/pretrained_attention_32f.h5'
    
    # 'models/weights_pretrained_autoencoder.h5'
    # tue issue might be that it isn't pretrained... hmmm....
    """
    train(batch_size                  = 64,
          test_interval               = 250,
          adversarial_samples         = 'x_fake',
          max_adversarial             = 0,
          d_thermostat                = 0.60,
          w_loss_re                   = 0.0, 
          w_loss_ammre                = 0.333,
          w_loss_perceptual           = 0.333,
          w_loss_rea                  = 0.333,
          num_batches                 = 50001,
          save_interval               = 25000,
          path_pretrained_autoencoder = 'models/pretrained_autoencoder_64px_32f_48z.h5',
          path_pretrained_attention   = None,
          save_path                   = 'nonadversarial_0_333_333_333'
         )
    """
