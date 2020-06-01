from glob import glob

import pandas as pd
import numpy as np

from skimage.io import imread
from cv2 import resize

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from vgg16_places_365 import VGG16_Places365



datagen = ImageDataGenerator(vertical_flip = True,
                             rotation_range = 45, )

image_info = pd.read_csv('first_test_cnn/image_classifications.csv')

train_generator = datagen.flow_from_dataframe(image_info, 
                                              directory = 'data/phenocam_images/',
                                              target_size = (224,224),
                                              batch_size = 32,
                                              x_col = 'file',
                                              y_col = 'crop_type',
                                              class_mode = 'raw',
                                              )


base_vgg_model = VGG16_Places365(weights='places', include_top = False)

train_features = base_vgg_model.predict_generator(train_generator)
train_y = to_categorical(image_info.crop_type.values)

# Setup a classification model to use on the features output from VGG16_places
# This is exactly the same as the top layer of VGG16, but with 5 classes instead of 365
model = keras.Sequential()
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.5, name='drop_fc1'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5, name='drop_fc2'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_features, train_y, validation_split=0.2, steps_per_epoch=10, epochs=100)
