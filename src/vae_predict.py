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
    ''' 
    Makes "n" predictions of new watch images. 
    Returns images as numpy array of size (n,128,128,3).'''

    # INPUT_DIM = (128,128,3)
        # # - - - SAME ARCHITECTURE USED IN TRAINING 
        # vae = VariationalAutoencoder( 
        #                 input_dim = INPUT_DIM
        #                 , encoder_conv_filters=[32,64,64,64,128,128]
        #                 , encoder_conv_kernel_size=[3,3,3,3,3, 3]
        #                 , encoder_conv_strides=[2,2,2,2,2, 2]
        #                 , decoder_conv_t_filters=[64,64,32,3,3, 3]
        #                 , decoder_conv_t_kernel_size=[3,3,3,3, 3, 3]
        #                 , decoder_conv_t_strides=[2,2,2,2, 2, 2]
        #                 , z_dim=200
        #                 , use_batch_norm=True
        #                 , use_dropout=True)


    vae=load_vae_model(model_id=model_id)
    znew = np.random.normal(size = (n_predictions,vae.z_dim))
    prediction = np.array(vae.decoder.predict(np.array(znew)))

    return prediction


def load_vae_model(model_id=16):
    # Clean the model_id from (int) to filepath
    model_id_clean = str('0000'+str(model_id))[-4:]

    # Unpickle the parameters for the model
    params = pickle.load(open('run/vae/'+model_id_clean+'_watches/params.pkl', 'rb'))
    
    # Initiate VAE model with new params
    vae = VariationalAutoencoder( 
                    input_dim = params[0]
                    , encoder_conv_filters=params[1]        #32,64,64,64,128,128]
                    , encoder_conv_kernel_size=params[2]    #[3,3,3,3,3, 3]
                    , encoder_conv_strides=params[3]        #[2,2,2,2,2, 2]
                    , decoder_conv_t_filters=params[4]      #[64,64,32,3,3, 3]
                    , decoder_conv_t_kernel_size=params[5]  #[3,3,3,3, 3, 3]
                    , decoder_conv_t_strides=params[6]      #[2,2,2,2, 2, 2]
                    , z_dim=params[7]                       #200
                    , use_batch_norm=params[8]              #True
                    , use_dropout=params[9])                #True)

    # load the weights for the model
    weights_file_path = 'run/vae/'+model_id_clean+'_watches/weights/weights.h5'
    vae.load_weights(weights_file_path)

    return vae 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_predictions', '-n', help='number of predictions to make', type=int)
    parser.add_argument('--model_id', '-m', help='pretrained model id to use for predictions', type=int)
    parser.add_argument('--save', '-s', help='save the image to the image_results directory',default=True)
    args = parser.parse_args()
    print('args: ', args)

    vae_result = vae_predict(args.n_predictions, args.model_id)
    vae_result_num = vae_result.shape[0]
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(vae_result[0])
    if args.save:
        # num_existing = str(len(os.listdir('../image_results/')))
        # name = '../image_results/VAE_prediction_'+num_existing+'.png'
        for img in range(vae_result_num):
            num_existing = str(len(os.listdir('../image_results/')))
            name = '../image_results/VAE_prediction_'+num_existing+'.png'
            img_i = vae_result[img]

        fig.savefig(name, dpi=125)

