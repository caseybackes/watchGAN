# from autoencoder import Autoencoder
import os
import argparse
from skimage import io, filters
import numpy as np
# from tensorflow import keras
import keras
from  keras.layers import Activation, Dense, Input
from  keras.layers import Conv2D, Flatten
from  keras.layers import Reshape, Conv2DTranspose
from  keras.models import Model, load_model
from  keras import backend as K
import tensorflow as tf


def create_denoise_ae(image_depth = 1000, epochs=300):
    # - - - DATA COLLECTION AND PREPROCESSING
    DATA_FOLDER = '../data/train/processed_images/' #contains the class dir of 'processed_images'

    data_clean = []
    data_noisy =[]

    for f in os.listdir(DATA_FOLDER)[0:image_depth]:
        fpath = os.path.join(DATA_FOLDER,f)
        img = io.imread(fpath)
        img = img/255.
        data_clean.append(img)
        noise = np.random.normal(loc=0.5, scale=0.5, size=img.shape)
        img_noisy = img + noise
        img_noisy = np.clip(img_noisy, 0., 1.) # all must be within [0,1] inclusive
        data_noisy.append(img_noisy)
    # Test/train split for AE validation
    train_depth = int(image_depth * 0.8)
    test_depth = image_depth-train_depth

    x_train_noisy, x_test_noisy = np.array(data_noisy[0:train_depth]), np.array(data_noisy[test_depth:-1])
    x_train, x_test = np.array(data_clean[0:train_depth]), np.array(data_clean[test_depth:-1])

    # - - - NETWORK PARAMETERS
    image_size = x_train[0].shape[0]
    input_shape = (image_size, image_size, 3)
    batch_size = 10
    kernel_size = 3
    latent_dim = 16 # our coding
    # Encoder/Decoder number of CNN layers and filters per layer
    # layer_filters = [16, 32]
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

    autoencoder.compile(loss='mse', optimizer='adam')
    print('Fitting data to autoencoder...')
    autoencoder.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy, x_test),
                    epochs=epochs,
                    batch_size=batch_size)

    # - - - SERIALIZE MODEL TO JSON
    print('Saving model...')
    model_to_json = autoencoder.to_json()

    # save new iterations of AE model
    how_many = sum(['ae_model' in x for x in os.listdir('run/ae')])+1
    name = 'run/ae/ae_model-'+str(how_many)+'.h5'
    autoencoder.save(name)
    print("Saved autoencoder model under ", name)
    return autoencoder
    

def load_ae_denoise(model_name, weights_name= None):
    '''
    model_name : (str) 
        Example: 'ae_model-1.json'
        These files are located in "run/ae/"
    weights_name : (str) 
        Example: 'ae_weights-1.h5'
        These files are located in "run/ae/"
    '''
    if '.json' in model_name and weights_name == None:
        print("A json model has no weights with it. Use the associated weights file or just the .h5 file.")
        return None 
    if weights_name:
        json_file = open('run/ae/'+model_name)
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into model 
        loaded_model.load_weights('run/ae/'+str(weights_name))
        return loaded_model
    else:
        return tf.keras.models.load_model('run/ae/'+str(model_name))


if __name__ == "__main__":

    # ARGPARSE FOR HYPERPERAMETERS
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--imdepth','-d', type=int, help='number of images to train model on', required=True)
    parser.add_argument('--epochs','-e', type = int, help='number of epochs for training',required=True)
    args = parser.parse_args()

    # GENERATE NEW AE MODEL 
    create_denoise_ae(image_depth=args.imdepth,epochs=args.epochs)