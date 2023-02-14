import numpy as np
from PIL import Image

arr = np.load("/home/postech2/irn/VOCdevkit/VOC2012/Divided/2007_000032_2_0_0_img.npy")

img = Image.fromarray(arr)
img.show()