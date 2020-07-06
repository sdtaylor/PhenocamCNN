import numpy as np

from keras_preprocessing.image import (ImageDataGenerator,
                                       load_img,
                                       img_to_array,
                                       )


def scale_images_0_1(x):
    x /= 127.5
    x -= 1
    return x 

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


def keras_predict(df, filename_col, model, target_size, preprocess_func,
                  image_dir=None, predict_prob=True, chunksize=500):
    """
    Load a keras model and predict on all images specified in the filename_col,
    of df

    """
    df = df.copy()
    df['class'] = 'a' # need a dummy class column to pass to the generator
    
    g  = ImageDataGenerator(preprocessing_function=preprocess_func).flow_from_dataframe(
                                         df, 
                                         directory = image_dir,
                                         target_size = target_size,
                                         batch_size = chunksize,
                                         shuffle = False,
                                         x_col = filename_col,
                                         y_col = 'class'
                                         )
    
    predictions = model.predict(g)
    
    if predict_prob:
        return predictions
    else:
        return np.argmax(predictions, 1)