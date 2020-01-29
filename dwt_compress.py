import os
import nibabel as nib
import pywt
import numpy as np
from info import *

def compressImgs(img_path, dwt_path, dwt_level=2, wavelet='db1'):
    if not os.path.exists(dwt_path):
        os.makedirs(dwt_path)
    for filename in sorted(os.listdir(img_path)):
        print(filename)
        img_etd = np.zeros([imgX, imgY, imgZ], dtype=img_dtype)
        filepath = os.path.join(img_path, filename)
        img = nib.load(filepath)
        affine = img.affine
        data = img.get_data()
        img_etd[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        data_lvln = pywt.wavedecn(data=img_etd, wavelet=wavelet, level=dwt_level)
        myimg = nib.Nifti1Image(data_lvln[0], affine)
        filepath_lvln = os.path.join(dwt_path, filename)
        nib.save(myimg, filepath_lvln)

    print("dwt{} image shape: {}".format(dwt_level, data_lvln[0].shape))

compressImgs(imgs_path_withfaces, imgs_path_withfaces_dwt, dwt_level=1)
compressImgs(imgs_path_nofaces, imgs_path_nofaces_dwt, dwt_level=1)


