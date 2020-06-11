from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#from vgg16_places_365 import VGG16_Places365


image_info = pd.read_csv('first_test_cnn/crop_only.csv')

# Setup one hot encoded class columns in the dataframe 
class_names = image_info.field_status_crop.unique()
class_names.sort()
class_names = ['class_'+str(c) for c in class_names]

one_hot_encoded = to_categorical(image_info.field_status_crop.values)

for name_i, n in enumerate(class_names):
    image_info[n] = one_hot_encoded[:,name_i]

########################################
# Setup validation split
validation_images = image_info.sample(frac=0.2)
train_images      = image_info[~image_info.index.isin(validation_images.index)]
train_images = train_images.sample(n=500, replace=True)


# n_classes = len(image_info.field_status_crop.unique())
# class_counts = image_info.groupby('field_status_crop').size().reset_index(name='counts')
# weights = {0:0 , 1:0.34, 2:0.98219, 3:0, 4:0.975}


image_dir = 'data/phenocam_images'
target_size = (48,48)
batch_size  = 32 # this batch size is just for extracting features
train_generator = ImageDataGenerator(vertical_flip = True,
                                     horizontal_flip = True,
                                     rotation_range = 45,
                                     #zoom_range = 0.25,
                                     width_shift_range = [-0.25,0,0.25],
                                     height_shift_range = [-0.25,0,0.25],
                                     shear_range = 45,
                                     #brightness_range = [0.2,1],
                                     fill_mode='reflect').flow_from_dataframe(
                                         train_images, 
                                         directory = image_dir,
                                         target_size = target_size,
                                         batch_size = batch_size,
                                         validation_split=0.2,
                                         x_col = 'file',
                                         y_col = class_names,
                                         class_mode = 'raw'
                                         )

# No random transformations for test images                                        
validation_generator  = ImageDataGenerator().flow_from_dataframe(
                                         validation_images, 
                                         directory = image_dir,
                                         target_size = target_size,
                                         batch_size = batch_size,
                                         x_col = 'file',
                                         y_col = class_names,
                                         class_mode = 'raw'
                                         )

# Example from https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg
base_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(48, 48, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.


# Freeze the base_model
base_model.trainable = False

original_weights = base_model.get_weights()

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(32, activation = 'relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(32, activation = 'relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(9,  activation = 'softmax')(x)

model = keras.Model(base_model.input, x)

model.compile(optimizer = keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])
model.fit(train_generator,
          validation_data= validation_generator,
          #class_weight = weights,
          epochs=5)
