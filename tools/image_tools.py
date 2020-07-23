import numpy as np
from keras_preprocessing.image import (load_img,
                                       img_to_array,
                                       )


# Load in images to numpy arrays
def load_imgs_from_df(df, x_col, img_dir, target_size, data_format='channels_last'):
    """
    From a dataframe load all images into a numpy array with final shape
    (n_images, height, width, 3), where height and width are specified in 
    target_shape.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe containing image info.
    x_col : str
        column in df which contains the image filenames.
    img_dir : str
        folder containing all the images.
    target_size : tuple
        final size to transform images to
    data_format : TYPE, optional
        DESCRIPTION. The default is 'channels_last'.

    Returns
    -------
    img_array : np.array
        array of images

    """
    
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
        