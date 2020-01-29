from helperFunction import *
from info import *

csImgs, csLabels = loadImages(imgs_path_withfaces, label=1)
ctImgs, ctLabels = loadImages(imgs_path_nofaces, label=0)
print(sizeof_fmt(csImgs))
print(sizeof_fmt(ctImgs))
