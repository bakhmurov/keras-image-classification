from keras.models import load_model
import numpy as np
from sys import argv
from classifier import Classifier

script, path = argv

cf = Classifier()
result = cf.classify(path)
print(result)