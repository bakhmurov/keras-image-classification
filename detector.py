import numpy as np
from keras.models import load_model
import numpy as np
from sys import argv
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
 
class Detector:
    def __init__(self):
        self.model = load_model('model.h5')
        self.image_size = self.model.input_shape[1]
        print("image size: " + str(self.image_size))
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = test_datagen.flow_from_directory('dataset/validate',
                                                        target_size = (self.image_size , self.image_size),
                                                        class_mode = 'categorical')
        self.labels = training_set.class_indices
        print(self.labels)

    def detect(self, imagePath): 
        test_image = image.load_img(imagePath, target_size = (self.image_size, self.image_size))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255.
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.model.predict_classes(test_image)
        return list(self.labels.keys())[result[0]];
