from vae_predict import vae_predict
from ae_denoise import load_unblur_model
import matplotlib.pyplot as plt 
# from autoencoder import Autoencoder
import argparse 
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

class Watch():
    ''' Instantiate with Watch(12)

    Current best vae_model_id=16
    Current best unblur_model_id=3
    
    
    '''
    def __init__(self
                , n_predictions=4
                , unblur_model_id=3
                , vae_model_id=16
                ):

        self.n_predictions = n_predictions

        # Convert from int to string location
        self.vae_model_id=vae_model_id # only needs the non-zero int in '0016_watches' dir name
        self.unblur_model_id='unblur_model'+str(unblur_model_id)+'.h5'
        
        # generate 'n' predictions
        self.predictions,self.vae_model = self.make_predictions()
        
        # Most effective deblur model is unblur_model3
        self.unblur_model = self.make_unblur_model()

        ## DEPRECITED ##
        #self.denoised_predictions = self.denoise_model.predict(self.predictions) 

        self.unblurred_predictions = self.unblur_model.predict(self.predictions) 
        self.prediction_shape = np.array(self.predictions[0].shape)

    def make_predictions(self):
        print('Makeing new images...')
        res = vae_predict(n_predictions=self.n_predictions
                            , model_id=self.vae_model_id)

        return res 

    def make_unblur_model(self):
        print('Loading the ', self.unblur_model_id,' autoencoder...')
        try:
            return load_unblur_model(self.unblur_model_id)
        except:
            print('Restoring the specified autoencoder didnt work... using default model "unblur_model4.h5"... ')
            return load_unblur_model('unblur_model5.h5')

    def generate(self):
        vae_results = self.make_predictions()



    def display(self, n_images=4, plotscale=3):
        '''
        n_rows (int): number of rows to plot in prediction image
        m_cols (int): number of cols to plot in the prediction image
        '''

        # rows, columns, and depth shape of whitespace
        #r,c,d = [int(x) for x in np.array(self.prediction_shape)*np.array([1,.20,1])]
        
        whitespace_rows = self.prediction_shape[0]#    -\
        whitespace_cols = self.prediction_shape[1]#      | -> for better readability than  
        whitespace_depth = self.prediction_shape[2]#   -/     the above list comprehension

        # Whitespace width between images can be less wide than a full image
        whitespace_cols = int(whitespace_cols*.2)
        whitespace = np.ones(shape=(whitespace_rows,whitespace_cols,whitespace_depth))
        
        # TODO: GET LATENT VECTOR AND RESHAPE INTO IMAGE ARRAY
        latent_vector = np.random.normal(size = (self.n_predictions,self.vae_model.z_dim))
        
        aspect_ratio = np.array([1,1.5])
        figsize= aspect_ratio*plotscale
        
        fig = plt.figure(figsize=tuple(figsize),dpi=120, clear=True)  
        plt.axis('off')
        for row in range(n_images): 
            ax1 = fig.add_subplot(4,3,row*3+1) 
            latim= latent_vector[row][0:195].reshape(13,15) 
            ax1.imshow(latim,cmap='gist_gray') 
            ax1.axis('off')
        for row in range(n_images): 
            ax2 = fig.add_subplot(4,3,row*3+2) 
            ax2.imshow(w.predictions[row]) 
            ax2.axis('off')
        for row in range(n_images):
            ax3 = fig.add_subplot(4,3,row*3+3)
            ax3.imshow(w.unblurred_predictions[row])
            ax3.axis('off')
        plt.tight_layout()
        plt.show()  




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_predictions','-n', type=int, help='number of images to train model on',default=4)
    parser.add_argument('--unblur_model_id','-u', type = int,help='number of epochs for training',default=3)
    parser.add_argument('--vae_model_id','-v', type = int,help='non-zero integer in the "0012_watches" dirname',default=16)
    parser.add_argument('--rows','-r',  help='rows in Watch().display(rows,cols,plotscale)', type=int)
    parser.add_argument('--cols','-c', help='cols in Watch().display(rows,cols,plotscale)', type=int)
    parser.add_argument('--plotscale','-p', help='scale size of plot in Watch().display(rows,cols,plotscale)')
    parser.add_argument('--save','-s', action='store_true', dest='save', help='opt to save the prediction plot image')
    args = parser.parse_args()

    w = Watch(n_predictions=args.n_predictions
            , unblur_model_id=args.unblur_model_id
            , vae_model_id=args.vae_model_id)

    w.display(4, 8)
    plt.show()