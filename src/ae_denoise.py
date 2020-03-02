# from autoencoder import Autoencoder
import os
import argparse
from skimage import io
from skimage import filters as skfilters
import numpy as np
import random
import keras
from  keras.layers import Activation, Dense, Input
from  keras.layers import Conv2D, Flatten
from  keras.layers import Reshape, Conv2DTranspose
from  keras.models import Model, load_model
from  keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
# REFERENCES: 
# https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian


def create_denoise_ae(image_depth=None, epochs=20, keep_model=False):
    """Generates new autoencoder model to trained to remove blur from images.

    Parameters
    ----------
    image_depth : int
        Number of images to train the model on, including train and test.

    epochs : int
        Number of epochs over which to train autoencoder model.

    keep_model : bool
        Flag to save the model or not. Default is `False`. 

    Returns
    -------
    numpy.ndarray
        An array of image arrays of shape (n,128,128,3) 
    """
    # Data collection and preprocessing
    DATA_FOLDER = '../data/train/processed_images/' #contains the class dir of 'processed_images'

    if image_depth == None:
        image_depth = -1

    X = []
    y = []

    # Retrieve all image file names availale
    all_images = os.listdir(DATA_FOLDER)
    
    # remove inappropriate file from list of images
    if '.DS_STORE' in all_images:
        all_images.remove('.DS_STORE')

    # shuffle the list of images (inplace)
    random.shuffle(all_images)

    # Apply the distortion (gaussian blur) to each image, saving a clean copy
    # and a distorted copy to X and y. 
    for f in all_images[0:image_depth]:
        if f == '.DS_STORE' or '.jpg' not in f:
            continue
        fpath = os.path.join(DATA_FOLDER,f)
        img = io.imread(fpath)

        # normalize images to a max value of 1
        img = img/255.

        # keep original clean image in y
        y.append(img)

        # Apply distortion (gaussian blur) and append to X. 
        # `multichannel=True` anticipates RGB (3 channel) images. 
        img_noisy = skfilters.gaussian(img, sigma=3, multichannel=True)  
        X.append(img_noisy)

    # Split the images for train and test. 
    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True,test_size=0.1, random_state=42)

    # convert each set to numpy.ndarray type for autoencoder
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # - - - NETWORK PARAMETERS
    image_size = x_train[0].shape[0]
    input_shape = (image_size, image_size, 3)
    batch_size = 10
    kernel_size = 3
    latent_dim = 16 # our coding
    # Encoder/Decoder number of CNN layers and filters per layer
    # layer_filters = [16, 32] #<- worked previously :) 
    layer_filters = [16,32,64,128]

    # - - - BUILD THE AUTOENCODER
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Stack of Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use MaxPooling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=2,
                activation='relu',
                padding='same')(x)

    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    # - - - BUILD THE DECODER
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)


    # Stack of Transposed Conv2D blocks
    # Notes:
    # 1) Use Batch Normalization before ReLU on deep networks
    # 2) Use UpSampling2D as alternative to strides>1
    # - faster but not as good as strides>1
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    x = Conv2DTranspose(filters=3,
                        kernel_size=kernel_size,
                        padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    # Compile and fit the Autoencoder Model
    autoencoder.compile(loss='mse', optimizer='adam')
    print('Fitting data to autoencoder...')
    autoencoder.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=True,
                    shuffle=True,
                    use_multiprocessing=True,)

    # Serialize and return the model 
    if keep_model:
        how_many = sum(['unblur_model' in x for x in os.listdir('run/ae')])+1
        name = 'run/ae/unblur_model'+str(how_many+1)+'.h5' # oddly, another plus one is needed for this to work as intended. 
        autoencoder.save(name)
        print("Saved autoencoder model under ", name)
    return autoencoder
    

def load_unblur_model(model_name):
    """Loads the specified autoencoder model including weights. 

    Parameters
    ----------
    model_name : str
        Exact name of the unblur model to use. Example: "unblur_model4.h5"

    Returns
    -------
    Model
        Fully trained and loaded autoencoder model ready for Model.predict() 
    """
    if type(model_name) ==  int:
        # get the clean model name from '4' to 'unblur_model4.h5'
        model_name = 'unblur_model'+str(model_name)+'.h5'

    return tf.keras.models.load_model('run/ae/'+model_name)


if __name__ == "__main__":

    # ARGPARSE FOR HYPERPERAMETERS
    parser = argparse.ArgumentParser(description='Train an unbluring autoencoder.')
    parser.add_argument('--imdepth','-d', type=int, default=100,
                        help='number of images to train model on. Default=100')
    parser.add_argument('--epochs','-e', type = int, required=True,
                        help='number of epochs for training')
    parser.add_argument('--save','-s', action='store_true', dest='save', 
                        help='include this opt flag to save the model')
    args = parser.parse_args()


    # Generate the deblurr model  
    create_denoise_ae(image_depth=args.imdepth ,
                      epochs=args.epochs ,
                      keep_model=args.save)