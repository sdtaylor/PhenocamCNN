from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from sklearn.utils.class_weight import compute_sample_weight

from vgg16_places_365 import VGG16_Places365


image_dir = './data/phenocam_images/'
target_size = (224,224)
validation_fraction = 0.2
train_sample_size = 2500
extract_batch_size  = 1000 # this batch size is just for extracting features

fit_batch_size = 10
fit_epoch_size = 500

assert fit_epoch_size % fit_batch_size == 0

image_info = pd.read_csv('first_test_cnn/crop_only.csv')

# Setup one hot encoded class columns in the dataframe 
class_names = image_info.field_status_crop.unique()
class_names.sort()
class_names = ['class_'+str(c) for c in class_names]

one_hot_encoded = to_categorical(image_info.field_status_crop.values)

for name_i, n in enumerate(class_names):
    image_info[n] = one_hot_encoded[:,name_i]

# train/validation split
validation_images = image_info.sample(frac=validation_fraction)
train_images      = image_info[~image_info.index.isin(validation_images.index)]
train_images['sample_weight'] = compute_sample_weight('balanced', train_images.field_status_crop)
train_images = train_images.sample(n=train_sample_size, replace=True, weights='sample_weight')


def scale_images(x):
    x /= 127.5
    x -= 1
    return x

train_generator = ImageDataGenerator(preprocessing_function=scale_images,
                                     vertical_flip = True,
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
                                         batch_size = extract_batch_size,
                                         shuffle = False,
                                         x_col = 'file',
                                         y_col = class_names,
                                         class_mode = 'raw'
                                         )

# No random transformations for test images                                        
validation_generator  = ImageDataGenerator(preprocessing_function=scale_images).flow_from_dataframe(
                                         validation_images, 
                                         directory = image_dir,
                                         target_size = target_size,
                                         batch_size = extract_batch_size,
                                         shuffle = False,
                                         x_col = 'file',
                                         y_col = class_names,
                                         class_mode = 'raw'
                                         )

#base_vgg_model = VGG16_Places365(weights='places', include_top = False)
base_vgg_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False,
) 


# hmm, potentially just do this process a bunch of times
# since the generator randomizes order and transformations.
# or just use the  keras.fit(class_weights) arge with this eq https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
train_features = base_vgg_model.predict_generator(train_generator)
train_y = train_images[class_names].values

validation_features = base_vgg_model.predict_generator(validation_generator)
validation_y = validation_images[class_names].values

class RandomImageGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, epoch_size):
        """
        The idea behind this is to extract the features of a bunch of randomlly generated
        images. Then feed those randomly to the fit method such that the same set
        is not seen on every epoch. 
        1. Have 500 training images to start with
        2. Sample with replacement to get 5000, pass all thru an ImageDataGenerator to
           get random transformations.
        3. Pass all 5000 thru the base VGG16 model to get features of all 5000.
        4. Pass the feature array here as x_set (and associated y_set for labels)
           epoch_size can be set to 1000, batch size to 50. Now each epoch will
           see 1000 random transformed images, but not the same 1000. 
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.total_n    = len(x_set)
        
        assert self.total_n > epoch_size, 'epoch size must be less than total x_set size'
        assert epoch_size % batch_size == 0, 'batch size should be multiple of epoch size'
        
        self.n_batches = int(self.epoch_size / self.batch_size)
        
        # initialize
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """
        Resample a new random batch.
        """
        self.epoch_indices = np.random.choice(np.arange(self.total_n), size=self.epoch_size)
        
    def __len__(self):
        """
        keras.model.fit expects this to be the number of batches, so its a fucntion
        of batch size.
        """
        return self.n_batches
    
    def __getitem__(self, idx):
        batch_idx = self.epoch_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x   = self.x[batch_idx]
        batch_y   = self.y[batch_idx]
        
        return (batch_x, batch_y)
        
train_feature_generator = RandomImageGenerator(train_features, train_y, 
                                               batch_size=fit_batch_size, epoch_size=fit_epoch_size)

# Setup a classification model to use on the features output from VGG16_places
# This is exactly the same as the top layer of VGG16, but with 5 classes instead of 365
model = keras.Sequential()
model.add(keras.layers.GlobalAveragePooling2D(name='flatten'))
model.add(Dense(512, activation='relu', name='fc1'))
model.add(Dropout(0.5, name='drop_fc1'))
model.add(Dense(512, activation='relu', name='fc2'))
model.add(Dropout(0.5, name='drop_fc2'))
model.add(Dense(len(class_names), activation='softmax'))
model.build(input_shape=base_vgg_model.output_shape)

model.compile(optimizer = keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])
print(model.summary())

model.fit(train_feature_generator, 
          validation_data = (validation_features, validation_y),
          #class_weight = weights,
          epochs=200)
