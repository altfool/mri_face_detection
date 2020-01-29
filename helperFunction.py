import numpy as np
import scipy.stats as stats
import os, sys
import nibabel as nib
from info import *

def loadImages(imgPath, label=0):
    # images with face features label 1, images without face features label 0;
    files = sorted(os.listdir(imgPath))
    imgs = np.zeros([len(files), imgX, imgY, imgZ], dtype=img_dtype)
    labels = np.zeros(len(files), dtype=np.int) + label
    for idx in range(len(files)):
        filepath = os.path.join(imgPath, files[idx])
        # print(filepath)
        data = nib.load(filepath).get_data()
        imgs[idx, :data.shape[0], :data.shape[1], :data.shape[2]] = data
    print("imgs shape: {}, labels shape: {}, label value: {}".format(imgs.shape, labels.shape, label))
    return imgs, labels


def sizeof_fmt(obj, suffix='B'):
    # check size of large object
    num = sys.getsizeof(obj)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)