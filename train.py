from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

image_size = 150

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (image_size, image_size),
                                                 batch_size = 6,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/validate',
                                            target_size = (image_size, image_size),
                                            batch_size = 6,
                                            class_mode = 'categorical')

labels = test_set.class_indices

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (image_size, image_size, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = len(labels), activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 4,
                         validation_data = test_set,
                         validation_steps = 10,
                         workers=16, 
                         use_multiprocessing=True
                         )

model.save("model.h5")
