from vae_predict import vae_predict, load_vae_model
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

        # PARAMETERS
        self.n_predictions = n_predictions
        self.unblur_model_id = unblur_model_id
        self.vae_model_id = vae_model_id

        # MODELS
        self.vae_model = load_vae_model(self.vae_model_id) 
        self.unblur_model = load_unblur_model(self.unblur_model_id)
        
        # PREDICTIONS
        self.vae_predictions = None
        self.unblurred_predictions = None
        self.prediction_shape = None 



    def generate(self):
        # generate predictions for the vae model and the unblur model
        self.vae_predictions = vae_predict(self.n_predictions,self.vae_model_id)#self.vae_model.decoder.predict(self.n_predictions)
        self.unblurred_predictions = self.unblur_model.predict(self.vae_predictions)

        # set attributes
        self.prediction_shape = self.unblurred_predictions[0].shape 

        # model_id_clean = str('0000'+str(self.vae_model_id))[-4:]
        # # model_name = model_id_clean+'_watches/'
        # # vae = tf.keras.models.load_model('run/vae/'+str(model_name))
        # model_file_path = 'run/vae/'+model_id_clean+'_watches/weights/weights.h5'
        # # vae.load_weights('run/vae/0004_watches/weights/weights.h5')
        # vae.load_weights(model_file_path)


    def display(self, n_images=4, plotscale=3, with_latent=False):
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
        
        # Create one columns of images at time
        fig = plt.figure(figsize=tuple(figsize),dpi=120, clear=True)  
        # plt.axis('off')

        for row in range(n_images): 
            ax1 = fig.add_subplot(n_images,3,row*3+1) 
            latim= latent_vector[row][0:195].reshape(13,15) 
            ax1.imshow(latim,cmap='gist_gray') 
            ax1.axis('off')
        for row in range(n_images): 
            ax2 = fig.add_subplot(n_images,3,row*3+2) 
            ax2.imshow(self.vae_predictions[row]) 
            ax2.axis('off')
        for row in range(n_images):
            ax3 = fig.add_subplot(n_images,3,row*3+3)
            ax3.imshow(self.unblurred_predictions[row])
            ax3.axis('off')
        plt.tight_layout()
        plt.show()  




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_predictions','-n', type=int, help='number of images to train model on',default=4)
    parser.add_argument('--unblur_model_id','-u', type = int,help='number of epochs for training',default=3)
    parser.add_argument('--vae_model_id','-v', type = int,help='non-zero integer in the "0012_watches" dirname',default=16)
    # parser.add_argument('--rows','-r',  help='rows in Watch().display(rows,cols,plotscale)', type=int)
    # parser.add_argument('--cols','-c', help='cols in Watch().display(rows,cols,plotscale)', type=int)
    parser.add_argument('--plotscale','-p', help='scale size of plot in Watch().display(rows,cols,plotscale)')
    parser.add_argument('--save','-s', action='store_true', dest='save', help='opt to save the prediction plot image')
    args = parser.parse_args()

    w = Watch(n_predictions=args.n_predictions
            , unblur_model_id=args.unblur_model_id
            , vae_model_id=args.vae_model_id)
    
    # Make predictions
    w.generate()
    
    
    # w.display(n_images=4, plotscale=8)
    # plt.show()

    fig = plt.figure()
    plt.suptitle("Single Unblur")

    for i in range(len(w.unblurred_predictions)):
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(w.unblurred_predictions[i])

    fig = plt.figure()
    plt.suptitle("Double Unblur")
    dbl_unblur = w.unblur_model.predict(w.unblurred_predictions)

    for i in range(len(dbl_unblur)):
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(dbl_unblur[i])
