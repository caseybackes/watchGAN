from vae_predict import vae_predict, load_vae_model
from ae_denoise import load_unblur_model
import matplotlib.pyplot as plt 
import argparse 
import os
from skimage import io, filters
import numpy as np
import keras
from  keras.layers import Activation, Dense, Input
from  keras.layers import Conv2D, Flatten
from  keras.layers import Reshape, Conv2DTranspose
from  keras.models import Model, load_model
from  keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 

class Watch():
    """The class object that can generate new images of watches. 
    Example: 
        >>> w = Watch(4)
        >>> w.generate()
        >>> w.display(n_images=4, plotscale=3)

    Notes: 
        - Current best vae_model_id=16
        - Current best unblur_model_id=3
    
    Attributes
    ----------
    n_predictions : int
        number of images to generate

    unblur_model_id : int
        "n" in unblur_modelN.h5. Defineds the unblur model to use. 
    
    ublur_model : Model
        Fully loaded Keras Model object of pretrained unblur autoencoder
    
    unblurred_predictions : numpy.ndarray
        Array of image arrays of shape (n, 128, 128, 3) given from the ublurring
        model where `n` is taken as the number of predictions made.
    
    vae_model_id : int
        non-zero unique integer in the directory list of 
        run/vae/
    
    vae_model : Model
        Fully loaded Keras Model object of pretrained variational autoencoder
    
    vae_predictions : numpy.ndarray
        Array of image arrays of shape (n, 128, 128, 3) where is is taken as
        the number of predictions made. 
    
    Methods
    -------
    generate(None)
        Generates n number of new images

    display(n_images=4, plotscale=3)
        Displays a figure of the newly generated images via matplotlib. 
    """
    
    def __init__(self, n_predictions=4, unblur_model_id=3, vae_model_id=16):

        # ATTRIBUTES
        self.n_predictions = n_predictions
        self.unblur_model_id = unblur_model_id
        self.vae_model_id = vae_model_id
        self.prediction_shape = None 


        # MODELS
        self.vae_model = load_vae_model(self.vae_model_id) 
        self.unblur_model = load_unblur_model(self.unblur_model_id)
        
        # PREDICTIONS
        self.vae_predictions = None
        self.unblurred_predictions = None


    def generate(self):
        """Generates new images of watches and returns a numpy.ndarray of 
        images of shape (n, 128,128,3). 

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            An array of image arrays of shape (n,128,128,3) 
        """

        # generate predictions for the vae model and the unblur model
        self.vae_predictions = vae_predict(self.n_predictions,self.vae_model_id)
        self.unblurred_predictions = self.unblur_model.predict(self.vae_predictions)

        # set shape attribute with latest information 
        self.prediction_shape = self.unblurred_predictions[0].shape 


    def display(self, n_images=4, plotscale=3):
        """Displays the generated images of watches and returns a figure. 

        Parameters
        ----------
        n_images : int
            number of images to display

        plotscale : int
            scaling factor for the size of the figure

        Returns
        -------
        matplotlib.pyplot.figure
            The figure object of the resulting display of images
        """

        # Put a white space between the images to display. 
        # Define the shape (rows, columns, and depth) of the whitespace  
        whitespace_rows = self.prediction_shape[0]
        whitespace_cols = self.prediction_shape[1]
        whitespace_depth = self.prediction_shape[2]
        
        # Whitespace width between images can be less wide than a full image
        whitespace_cols = int(whitespace_cols*.2)
        whitespace = np.ones(shape=(whitespace_rows,
                                    whitespace_cols,
                                    whitespace_depth))
        
        latent_vector = np.random.normal(size = (self.n_predictions,
                                                 self.vae_model.z_dim))
        
        # The size of the figure can be adjusted while 
        # respecting the aspect ratio 
        aspect_ratio = np.array([1,1.5])
        figsize = aspect_ratio*plotscale
        
        # Create one column of the plot at time, starting with the latent
        # vector in the first column, then the vae img, then the unblur img. 
        fig = plt.figure(figsize=tuple(figsize),dpi=120)  

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
        return fig 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate new watch designs.')
    parser.add_argument('--n_predictions','-n', type=int, 
                            help='number of images to train model on',
                            default=4)
    parser.add_argument('--unblur_model_id','-u', type = int,
                            help='number of epochs for training',
                            default=3)
    parser.add_argument('--vae_model_id','-v', type = int,
                            help='non-zero integer in the "0012_watches" dirname',
                            default=16)
    parser.add_argument('--plotscale','-p', type=int,
                            help='scale size of plot when displaying results')
    parser.add_argument('--save','-s', action='store_true', dest='save', 
                            help='opt flag to save the prediction plot image')
    args = parser.parse_args()

    # Intitate the class object
    w = Watch(n_predictions=args.n_predictions, 
              unblur_model_id=args.unblur_model_id, 
              vae_model_id=args.vae_model_id)
    # Generate new images
    w.generate()

    # Show the results
    w.display(4, 8)
    plt.show()