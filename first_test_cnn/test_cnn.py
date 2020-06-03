from glob import glob

import pandas as pd
import numpy as np

from skimage.io import imread
from cv2 import resize
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from vgg16_places_365 import VGG16_Places365


image_info = pd.read_csv('first_test_cnn/image_classifications.csv')
validation_images = image_info.sample(frac=0.2)
train_images      = image_info[~image_info.index.isin(validation_images.index)]
train_images = train_images.sample(n=1000, replace=True)


n_classes = len(image_info.crop_type.unique())
class_counts = image_info.groupby('crop_type').size().reset_index(name='counts')
weights = {0:0 , 1:0.34, 2:0.98219, 3:0, 4:0.975}


target_size = (32,32)
batch_size  = 32
train_generator = ImageDataGenerator(vertical_flip = True,
                                     rotation_range = 45).flow_from_dataframe(
                                         train_images, 
                                         directory = 'data/phenocam_images/',
                                         target_size = target_size,
                                         batch_size = batch_size,
                                         x_col = 'file',
                                         y_col = 'crop_type',
                                         class_mode = 'raw'
                                         )

# No random transformations for test images                                        
validation_generator  = ImageDataGenerator().flow_from_dataframe(
                                         validation_images, 
                                         directory = 'data/phenocam_images/',
                                         target_size = target_size,
                                         batch_size = batch_size,
                                         x_col = 'file',
                                         y_col = 'crop_type',
                                         class_mode = 'raw'
                                         )

base_vgg_model = VGG16_Places365(weights='places', include_top = False)
# hmm, potentially just do this process a bunch of times
# since the generator randomizes order and transformations.
# or just use the  keras.fit(class_weights) arge with this eq https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
train_features = base_vgg_model.predict_generator(train_generator)
train_y = to_categorical(train_images.crop_type.values)

validation_features = base_vgg_model.predict_generator(validation_generator)
validation_y = to_categorical(validation_images.crop_type.values)

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
                                               batch_size=25, epoch_size=100)

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
model.fit(train_feature_generator, 
          validation_data = (validation_features, validation_y),
          class_weight = weights,
          epochs=100)
