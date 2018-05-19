import numpy as np
from keras.models import load_model
import numpy as np
from sys import argv
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
 
class Detector:
    def __init__(self):
        self.image_size = 150
        self.model = load_model('model.h5')
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = test_datagen.flow_from_directory('dataset/validate',
                                                        target_size = (self.image_size , self.image_size),
                                                        batch_size = 6,
                                                        class_mode = 'categorical')
        self.labels = training_set.class_indices
        print(self.labels)

    def detect(self, imagePath): 
        test_image = image.load_img(imagePath, target_size = (self.image_size, self.image_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.model.predict(test_image)
        found = ""
        for i, res in enumerate(result[0]):
            if res == 1:
                if found != "":
                    found += ","
                found += list(self.labels.keys())[i]
        print(result[0])
        return None if found == "" else found
