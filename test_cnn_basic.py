from glob import glob
import re

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from keras_preprocessing.image import (ImageDataGenerator,
                                       load_img,
                                       img_to_array,
                                       )
from tensorflow.keras.utils import to_categorical

from sklearn.utils.class_weight import compute_sample_weight

from vgg16_places_365 import VGG16_Places365


image_dir = 'data/phenocam_images/'
train_sample_size = 20000

validation_fraction = 0.2
target_size = (224,224)
batch_size  = 128

image_info = pd.read_csv('first_test_cnn/crop_only.csv')

print('sample sizes: {s}'.format(s=image_info.field_status_crop.value_counts()))


# Setup one hot encoded class columns in the dataframe 
class_names = image_info.field_status_crop.unique()
class_names.sort()
class_names = ['class_'+str(c) for c in class_names]

one_hot_encoded = to_categorical(image_info.field_status_crop.values)

for name_i, n in enumerate(class_names):
    image_info[n] = one_hot_encoded[:,name_i]

########################################
# Setup validation split
total_validation_images = int(len(image_info) * validation_fraction)

# First put all images from held out sites into the validation set
image_info['validation_site'] = image_info.file.apply(lambda f: bool(re.search(r'(arsmorris2)|(mandani2)|(cafboydnorthltar01)', f)))
validation_images = image_info[image_info.validation_site]

# Add in a random sample of remaining images to get to the total validation fraction
# images from validation sites excluded here by setting weight to 0
image_info['validation_weight'] = image_info.validation_site.apply(lambda val_site: 0 if val_site else 1)
validation_images = validation_images.append(image_info.sample(n= total_validation_images - len(validation_images), replace=False, weights='validation_weight')) 

# Training images are ones that are left
train_images = image_info[~image_info.index.isin(validation_images.index)]

# assure no validtion sites in the training data, and all images in each set are unique
assert train_images.validation_site.sum() == 0, 'validation sites in training dataframe'
assert train_images.index.nunique() == len(train_images), 'duplicates in training dataframe'
assert validation_images.index.nunique() == len(validation_images), 'duplicates in validation dataframe'

# expand training by random sampling, weighted so that low sample size category images are repeated.
# This makes it so sample sizes are even in training
train_images['sample_weight'] = compute_sample_weight('balanced', train_images.field_status_crop)
train_images = train_images.sample(n=train_sample_size, replace=True, weights='sample_weight')


# Load in images to numpy arrays
def load_imgs_from_df(df, x_col, img_dir, target_size, data_format='channels_last'):
    n_images = len(df)
    img_array = np.zeros((n_images,) + target_size + (3,))
    
    for i,j in enumerate(df[x_col]):
        img = load_img(img_dir + j,
                       color_mode='rgb',
                       target_size=target_size)
        img_array[i] = img_to_array(img, data_format=data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
    
    return img_array
    

train_x = load_imgs_from_df(train_images, x_col='file', img_dir=image_dir, target_size=target_size, data_format='channels_last')
train_y = train_images[class_names].values

val_x = load_imgs_from_df(validation_images, x_col='file', img_dir=image_dir, target_size=target_size, data_format='channels_last')
val_y = validation_images[class_names].values

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
                                     fill_mode='reflect').flow(
                                         x = train_x,
                                         y = train_y,
                                         shuffle = True,
                                         batch_size = batch_size,
                                         )

# No random transformations for test images                                        
validation_generator  = ImageDataGenerator(preprocessing_function=scale_images).flow(
                                         x = val_x,
                                         y = val_y,
                                         shuffle = True,
                                         batch_size = batch_size,
                                         )

# Example from https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg
#base_model = VGG16_Places365(
base_model = keras.applications.VGG16(
    weights=None,  # Load weights pre-trained on ImageNet.
    input_shape= target_size + (3,),
    include_top=True,
    classes=len(class_names)
)  # Do not include the ImageNet classifier at the top.


# Freeze the base_model
#base_model.trainable = False

#original_weights = base_model.get_weights()

#x = base_model.output
#x = keras.layers.GlobalMaxPooling2D()(x)
#x = keras.layers.Dense(128, activation = 'relu')(x)
#x = keras.layers.Dropout(0.5)(x)
#x = keras.layers.Dense(128, activation = 'relu')(x)
#x = keras.layers.Dropout(0.5)(x)
#x = keras.layers.Dense(len(class_names),  activation = 'softmax')(x)

#model = keras.Model(base_model.input, x)

base_model.compile(optimizer = keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])
print(base_model.summary())

base_model.fit(train_generator,
          validation_data= validation_generator,
          #class_weight = weights,
          epochs=30)
