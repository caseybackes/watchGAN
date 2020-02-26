from vae_predict import vae_predict
from ae_denoise import load_ae_denoise
import matplotlib.pyplot as plt 
# from autoencoder import Autoencoder
import os
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

class Watch():
    ''' Instantiate with Watch()'''
    def __init__(self,n_predictions=10):
        self.predictions = self.make_predictions(n=n_predictions)
        self.denoise_model = self.make_denoise_model('ae.h5')
        self.denoised_predictions = self.denoise_model.predict(self.predictions) 

    def make_predictions(self,n=10):
        print('Makeing new images...')
        return vae_predict(n)

    def make_denoise_model(self,model_name):
        print('Loading the denoising autoencoder...')
        try:
            lad = load_ae_denoise(model_name)
            return lad 
        except:
            print('Restoring the denoising autoencoder didnt work... giving you the default "ae.h5"... ')
            lad = load_ae_denoise('ae.h5')
            return lad

    def __repr__(self):
        return "<Watch: has_predictions:("('T' if self.predictions else "F")+ ");been_denoised:("+('T' if self.denoised_predictions else 'F')+ ">"

    # # MAKE A PREDICTIO FROM THE DECODER OF THE VAE. 
    # print('getting predictions')
    # result = vae_predict(10)

    # # DENOISE THE PREDICTION
    # print('Reviving the denoising autoencoder...')
    # # breakpoint()
    # denoising_ae = load_ae_denoise('ae.h5') 

    # print('"denoising" predictions')
    # processed_predictions = denoising_ae.predict(result)

    # for i in range(len(processed_predictions)):
    #     plt.imshow(processed_predictions[i])


