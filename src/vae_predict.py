import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
import os
from scipy.stats import norm
import pandas as pd
import argparse 
from vae.VAE_MODEL import VariationalAutoencoder
from skimage.color import rgb2hsv
# REFERENCES: 
# https://blog.keras.io/building-autoencoders-in-keras.html

def vae_predict(n_predictions, model_id=4):
    """Generates new images of watches and returns a numpy.ndarray of 
    images of shape (n, 128,128,3). 

    Parameters
    ----------
    n_predictions : int
        number of new images to generate

    model_id : int
        non-zero unique integer in the directory list of run/vae/

    Returns
    -------
    numpy.ndarray
        An array of image arrays of shape (n,128,128,3) 
    """

    # Load pretrained model
    vae=load_vae_model(model_id=model_id)

    # Define new values for each dimension of the latent vector 
    # from a normal random distribution
    znew = np.random.normal(size = (n_predictions,vae.z_dim))

    # Use the new latent vector to generate a preidction (new image)
    prediction = np.array(vae.decoder.predict(np.array(znew)))

    # Return an array of arrays with each representing an image array. 
    return prediction


def load_vae_model(model_id=16):
    """Loads a previously trained variational autoencoder model

    Parameters
    ----------
    model_id : int
        non-zero unique integer in the directory list of run/vae/

    Returns
    -------
    Model
        Fully trained variational autoencoder of predefined architecture and 
        associated weights for that model.
    """

    # Clean the model_id from (int) to filepath
    model_id_clean = str('0000'+str(model_id))[-4:]

    # Unpickle the parameters for the model
    params = pickle.load(
            open('run/vae/'+model_id_clean+'_watches/params.pkl', 'rb'))
    
    # Initiate VAE model with new params
    vae = VariationalAutoencoder(                           # EXAMPLE:
                    input_dim = params[0]                   #(128,128,3)
                    , encoder_conv_filters=params[1]        #[32,64,64,64,128,128]
                    , encoder_conv_kernel_size=params[2]    #[3,3,3,3,3, 3]
                    , encoder_conv_strides=params[3]        #[2,2,2,2,2, 2]
                    , decoder_conv_t_filters=params[4]      #[64,64,32,3,3, 3]
                    , decoder_conv_t_kernel_size=params[5]  #[3,3,3,3, 3, 3]
                    , decoder_conv_t_strides=params[6]      #[2,2,2,2, 2, 2]
                    , z_dim=params[7]                       #200
                    , use_batch_norm=params[8]              #True
                    , use_dropout=params[9])                #True)

    # Find the weights file of interest
    weights_file_path = 'run/vae/'+model_id_clean+'_watches/weights/weights.h5'

    # Load the weights and return the fully ready model. 
    vae.load_weights(weights_file_path)
    return vae 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_predictions', '-n', type=int,
                        help='number of predictions to make')
    parser.add_argument('--model_id', '-m', type=int,
                        help='pretrained model id to use for predictions')
    parser.add_argument('--save', '-s', default=True,
                        help='save the image to the image_results directory')
    args = parser.parse_args()
    print('args: ', args)

    # Us the above parameters to generate a new image
    vae_result = vae_predict(args.n_predictions, args.model_id)

    vae_result_num = vae_result.shape[0]

    # Display the first resulting newly generated image. 
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(vae_result[0])

    if args.save:
        for img in range(vae_result_num):
            num_existing = str(len(os.listdir('../image_results/')))
            name = '../image_results/VAE_prediction_'+num_existing+'.png'
            img_i = vae_result[img]

        fig.savefig(name, dpi=125)

