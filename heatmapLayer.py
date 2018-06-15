from keras.models import load_model
import numpy as np
from sys import argv
from layerOutput import LayerOutput
import matplotlib.pyplot as plt

script, path = argv

lo = LayerOutput()
result = lo.extract(path, "average_pooling2d_1")[0]
im = plt.imread(path)
implot = plt.imshow(im)
extent = implot.get_extent()
plt.imshow(result[:,:,3], cmap='hot', alpha=.4, interpolation='nearest', extent=extent)
plt.show()
#plt.savefig(path.replace(".jpg","_heatmap.jpg"))