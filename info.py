import numpy as np

img_dtype = np.float32

imgX, imgY, imgZ = (256, 256, 150)
imgs_path_withfaces = '../dataset/withfaces'
imgs_path_nofaces = '../dataset/nofaces'

imgX_dwt1, imgY_dwt1, imgZ_dwt1 = (128, 128, 75)
imgs_path_withfaces_dwt = './dataset/withfaces'
imgs_path_nofaces_dwt = './dataset/nofaces'

dwt_flag = (True, False)[0]
if dwt_flag:
    imgX, imgY, imgZ = imgX_dwt1, imgY_dwt1, imgZ_dwt1
    imgs_path_withfaces = imgs_path_withfaces_dwt
    imgs_path_nofaces = imgs_path_nofaces_dwt
