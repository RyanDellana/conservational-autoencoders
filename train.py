from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

#if len(sys.argv) < 2:
#    print('please specify <gpu_number>')
#    exit()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pickle
import cv2

from models import Composite

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


def train(save_path,
          batch_size                   = 64,
          max_epochs                   = 100,
          w_loss_output_conserv_person = 0.0,
          w_loss_output_conserv_car    = 0.0,
          w_loss_fakeness              = 0.25,
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
    # Add third "fake" channel to the y labels.
    y_train = np.concatenate((y_train, np.ones((y_train.shape[0],64,64,1))*-1), axis=-1)
    y_val   = np.concatenate((y_val, np.ones((y_val.shape[0],64,64,1))*-1), axis=-1)
    # build composite model for training autoencoder
    args = {'output_cnls':3, 
            'alpha':1.0, 
            'name':'composite', 
            'encode_segmenter_output':encode_segmenter_output,
            'encode_segmenter_skips':encode_segmenter_skips
           }
    print('building models...')
    composite = Composite(**args)
    segmenter, ae, encoder, decoder = composite.segmenter, composite.ae, composite.encoder, composite.decoder
    batches_per_epoch = x_train.shape[0] // batch_size
    seg_best, composite_best = 10000.0, 10000.0
    print('Define optimizers...')
    optimizer_composite = Adam(0.001) #Adam(0.0001, 0.5)
    optimizer_seg       = Adam(0.001) #Adam(0.0001, 0.5)
    print('training...')
    for idx_epoch in range(max_epochs):
        print('epoch', idx_epoch)
        x_train, y_train = shuffle_in_unison(x_train, y_train)
        for idx_batch in range(100): #range(x_train.shape[0] // batch_size):
            print('batch', idx_batch)
            from_, to_ = idx_batch*batch_size, (idx_batch+1)*batch_size
            x_batch, y_batch = x_train[from_:to_], y_train[from_:to_]
            #print("x_batch.shape =", x_batch.shape)
            #print('composite.predict for adversarial augmentation...')
            """
            x_fake, seg_x, seg_x_fake = composite([x_batch[0:16], y_batch[0:16]])
            loss_re, loss_output_conserv_person = composite.losses[0:2]
            loss_output_conserv_car, loss_fakeness, iou_seg_x_seg_x_fake = composite.losses[2:5]
            iou_seg_g_seg_x, iou_seg_g_seg_x_fake = composite.losses[5:7]
            """
            #print('loss_feature_conserv =', np.mean(loss_feature_conserv))
            #print('loss_deep_feature_conserv =', np.mean(loss_deep_feature_conserv))
            # Replace 2 images with fake versions
            pass
            # Use the other 14 to augment their images with a random blend. (i.e. horizontal or vertical line merge)
            pass
            # Update the hallucination channel of the y_batch to reflect the augmentation
            pass
            # train models on the batch
            #print('segmenter.train_on_batch...')
            with tf.GradientTape() as tape:
                seg_x, seg_shallow_skips, seg_deep_skips = segmenter(x_batch)
                loss = tf.keras.losses.MeanSquaredError()(seg_x, y_batch)
                segmenter.trainable = True
                grads = tape.gradient(loss, segmenter.trainable_weights)
                optimizer_seg.apply_gradients(zip(grads, segmenter.trainable_weights))
                segmenter.trainable = False
                print('segmenter loss =', np.mean(loss))
            #print('composite.train_on_batch...')
            with tf.GradientTape() as tape:
                segmenter.trainable = False
                x_fake, seg_x_raw, seg_x_raw_fake = composite([x_batch, y_batch])
                loss_re, loss_output_conserv_person = composite.losses[0:2]
                loss_output_conserv_car, loss_fakeness, iou_seg_x_seg_x_fake = composite.losses[2:5]
                iou_seg_g_seg_x, iou_seg_g_seg_x_fake = composite.losses[5:7]
                loss_composite = loss_re*0.5 + loss_output_conserv_person*0.25 + loss_output_conserv_car*0.25
                grads = tape.gradient(loss_composite, composite.trainable_weights)
                optimizer_composite.apply_gradients(zip(grads, composite.trainable_weights))
                print('loss_re =', np.mean(loss_re))
            
        sample_imgs_x = x_val[0:200, :, :, :]
        seg_x, seg_shallow_skips, seg_deep_skips = segmenter(sample_imgs_x)
        seg_x_cars = np.tile(seg_x[:, :, :, 1:2], 3) # grab one of the labels and tile it.
        save_images(path  = save_path+'/images/cars'+str(idx_epoch)+'.jpg', 
                    imgs0 = list(sample_imgs_x), 
                    imgs1 = list(seg_x_cars))
        ae_prediction = ae(sample_imgs_x)
        ae_prediction = np.array(ae_prediction, dtype=np.float32)
        save_images(path  = save_path+'/images/ae'+str(idx_epoch)+'.jpg',
                    imgs0 = list(sample_imgs_x),
                    imgs1 = list(ae_prediction))

        # TODO print(idx_epoch, len(adv_imgs), loss_re, loss_ammre, loss_attn, iou_mean_x, iou_mean_fake)
        # Test the segmenter on the validation set
        #seg_acc, seg_loss = segmenter.evaluate(x=x_val, y=y_val) # TODO verify.
        # Evaluate the composite model somehow... i.e. the autoencoder...
        """
        pass
        if seg_loss < seg_best:
            segmenter.save_weights(save_path+'/model_saves/segmenter_weights.h5')
            seg_best = seg_loss
            # if comp_loss < composite_best: # TODO Doesn't seem like a good metric. TODO
            encoder.save_weights(save_path+'/model_saves/encoder_weights.h5')
            decoder.save_weights(save_path+'/model_saves/decoder_weights.h5')
            composite_best = comp_loss
        x_test_batch = x_val[0:64]
        y_test_batch = y_val[0:64]
        ret = composite.predict([x_test_batch, y_test_batch])
        x_fake, seg_x, seg_x_fake = ret[0:3] 
        loss_re, loss_feature_conserv, loss_deep_feature_conserv = ret[3:6]
        loss_output_conserv_person, loss_output_conserv_car, loss_fakeness = ret[6:9]
        loss_kl, iou_seg_x_seg_x_fake, iou_seg_g_seg_x, iou_seg_g_seg_x_fake = ret[9:]
        #x_fake_cv2 = (x_fake*255.0).astype(np.uint8)
        #seg_img
        #save_images(save_path+'/images/'+str(idx_epoch)+'_fake.jpg', x_test_batch, x_fake)
        #save_images(save_path+'/images/'+str(idx_epoch)+'_seg_person.jpg', x_test_batch, seg_x[])
        #save_images(save_path+'/images/'+str(idx_epoch)+'_seg_car.jpg', x_test_batch, seg_x[])
        """

# IDEA: Segmenter has a "fake" channel.
# How do we label fake samples? Car + person + fake? or just fake?
# How strong is the penalty for fake?
# Do we mix parts of images so half an image is real and half is fake or something?
# Some images will be all real or all fake, and others will have a random rectangular region that's either real or fake.
# Choose either a random horizontal line or a random vertical line within a certain range.
# TODO TODO ^^^^
if __name__ == '__main__':
    
    train(save_path                    = 'ae_seg_05_025_025_transposed_seg',
          batch_size                   = 64,
          max_epochs                   = 100,
          w_loss_output_conserv_person = 0.0,
          w_loss_output_conserv_car    = 0.0,
          w_loss_fakeness              = 0.25,
          encode_segmenter_output      = False,
          encode_segmenter_skips       = False,
          path_pretrained_segmenter    = None,
          path_pretrained_encoder      = None,
          path_pretrained_decoder      = None)



