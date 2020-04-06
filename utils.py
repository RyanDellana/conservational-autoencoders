import math
import numpy as np
import cv2


def cv2_2_keras(imgs_cv2):
    """ imgs_cv2 can be a single image or a list of images. """
    if type(imgs_cv2) is list:
        imgs_keras = []
        for img in imgs_cv2:
            img_ = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgs_keras.append(img_.astype(np.float32)/127.5 - 1.0)
        return np.array(imgs_keras, dtype=np.float32)
    else:
        img = imgs_cv2
        img_ = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img_.astype(np.float32)/127.5 - 1.0


# TODO doesn't print attention images properly because they have sigmoid outputs.
def keras_2_cv2(imgs_keras):
    """ imgs_keras can be a single image, a list of images, or a numpy array of images. """
    if (type(imgs_keras) is list) or (type(imgs_keras) is np.ndarray and len(imgs_keras.shape) == 4):
        imgs_cv2 = [((img + 1.0) * 127.5).astype(np.uint8) for img in imgs_keras]
        if imgs_cv2[0].shape[2] == 1: # if single channel
            imgs_cv2 = [cv2.cvtColor(img.reshape((64, 64)), cv2.COLOR_GRAY2BGR) for img in imgs_cv2]
        return np.array(imgs_cv2, dtype=np.uint8)
    else:
        img_cv2 = ((imgs_keras + 1.0) * 127.5).astype(np.uint8)
        if img_cv2.shape[2] == 1: # if single channel
            img_cv2 = img_cv2.reshape((64, 64))
            img_cv2 = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_cv2]
        return img_cv2


def save_images(path, imgs0, imgs1):
    """ save a composite image showing each img in imgs1 next to each corresponding img in imgs0 """
    assert len(imgs0) in [50, 200]
    assert len(imgs1) in [50, 200]
    imgs0_cv2 = keras_2_cv2(imgs0)
    imgs1_cv2 = keras_2_cv2(imgs1)
    res = 64
    num_rows = 5 if len(imgs0) == 50 else 10
    img_reconstructions = np.zeros((res*num_rows*2,res*num_rows*2,3), dtype=np.uint8)
    for idx, (img0, img1) in enumerate(zip(imgs0_cv2, imgs1_cv2)):
        row, col = int(idx // num_rows), int(idx % num_rows * 2)
        img_reconstructions[  row*res:(row+1)*res,  col*res:(col+1)*res , :] = img0[:,:,:]
        img_reconstructions[  row*res:(row+1)*res,  (col+1)*res:(col+2)*res, :] = img1[:,:,:]
    cv2.imwrite(path, img_reconstructions)


def resize_pad(img, dim, interp=None):
    """ resize and pad a cv2 image so that it has a length and width given by ``dim`` """
    h, w = img.shape[0:2]
    img_zeros = np.zeros((dim, dim, 3) if len(img.shape) == 3 else (dim, dim), np.uint8)
    resize_percent = float(dim) / max(h, w)
    w_, h_ = int(math.ceil(w*resize_percent)), int(math.ceil(h*resize_percent))
    if interp is None:
        img_resized = cv2.resize(img, (w_, h_))
    else:
        img_resized = cv2.resize(img, (w_, h_), interpolation=interp)
    img_zeros[(dim-h_)/2:(dim-h_)/2+h_, (dim-w_)/2:(dim-w_)/2+w_] = img_resized[:,:]
    return img_zeros



