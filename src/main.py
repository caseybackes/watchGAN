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
import matplotlib.pyplot as plt 

# TODO: RETRAIN VAE MODEL ON CLEANED DATASET


class Watch():
    ''' Instantiate with Watch()'''
    def __init__(self,n_predictions=10,unblur_model_id=4, vae_model_id=4):
        self.n_predictions = n_predictions
        self.vae_model_id=vae_model_id
        self.unblur_model_id=unblur_model_id 
        self.predictions = self.make_predictions(n=self.n_predictions,)
        # latest and greatest deblur
        self.ublur_model = self.make_unblur_model('unblur_model'+str(unblur_model_id)+'.h5')
        self.denoised_predictions = self.denoise_model.predict(self.predictions) 
        self.prediction_shape = np.array(self.predictions[0].shape)

    def make_predictions(self,n=10, model_id = vae_model_id):
        print('Makeing new images...')
        return vae_predict(n)

    def make_unblur_model(self,model_name):
        print('Loading the unblur autoencoder...')
        try:
            lad = load_ae_denoise(model_name)
            return lad 
        except:
            print('Restoring the denoising autoencoder didnt work... giving you the default "ae.h5"... ')
            lad = load_ae_denoise('ae.h5')
            return lad

    def display(self):
        rows = 5
        cols = 5
        n_to_show = self.predictions.shape[0]

        r,c,d = [int(x) for x in np.array(self.prediction_shape)*np.array([1,.5,1])]
        whitespace = np.ones(shape=(r,c,d))
        fig = plt.figure(figsize=(15,10))
        for i in range(n_to_show):
            ax = fig.add_subplot(rows, cols, i+1)
            side_by_side = np.concatenate( (self.predictions[i],whitespace,self.denoised_predictions[i]) , axis = 1)
            ax.imshow(side_by_side)
            ax.axis('off')
        plt.tight_layout(pad=1)

w = Watch(20)
w.display()
plt.show()