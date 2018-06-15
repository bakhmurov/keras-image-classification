from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

image_size = 64

train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   #rotation_range=100, 
                                   #width_shift_range=0.1,
                                   #height_shift_range=0.1, 
                                   #fill_mode='nearest',
                                   #vertical_flip = True,
                                   #horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 shuffle = True,
                                                 target_size = (image_size, image_size),
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/validate',
                                            target_size = (image_size, image_size),
                                            class_mode = 'categorical')

labels = test_set.class_indices

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape = (image_size, image_size, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = len(labels), activation = 'softmax'))

sgd = optimizers.SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
history = model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(training_set,
                         epochs = 50,
                         shuffle = True,
                         validation_data = test_set,
                         workers=16, 
                         use_multiprocessing=True
                         )

model.save("model.h5")
