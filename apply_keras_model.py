import glob
from os.path import basename

import pandas as pd

from tensorflow import keras
import keras_tools

results_file = './results/vgg16_predictions.csv'

class_categories = pd.read_csv('classification_categories_crop_only.csv')
keras_model = keras.models.load_model('./fit_models/vgg16_90epochs.h5')

image_dir = '/project/ltar_phenology_proj1/data/phenocam/images/'
#image_dir = './data/phenocam_images'

all_images = glob.glob(image_dir+ '**/*.jpg', recursive=True)

image_info = pd.DataFrame(dict(filepath = all_images))

image_info['file'] = image_info.applymap(basename)

for c in class_categories.class_description:
    image_info[c] = 0
    
predictions = keras_tools.keras_predict(df = image_info, filename_col='filepath', 
                                        model = keras_model, target_size=(224,224),
                                        preprocess_func=keras_tools.scale_images_0_1)

image_info.loc[:,class_categories.class_description] = predictions.round(5)

image_info.to_csv(results_file, index=False)
