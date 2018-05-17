from keras.models import load_model
import numpy as np
from sys import argv
from detector import Detector

script, path = argv

det = Detector()
result = det.detect(path)
print(result)