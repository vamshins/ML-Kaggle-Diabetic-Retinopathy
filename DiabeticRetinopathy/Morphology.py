__author__ = 'Vamshi'
import matplotlib.pyplot as plt
from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage import io

phantom = img_as_ubyte(io.imread('C:/Users/Vamshi/Anaconda/lib/site-packages/skimage/data/astronaut.png', as_grey=True))
fig, ax = plt.subplots()
ax.imshow(phantom, cmap=plt.cm.gray)