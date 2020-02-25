import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import pandas as pd
import argparse 
from vae.VAE_MODEL import VariationalAutoencoder
from utils.loaders import load_model, ImageLabelLoader
from datetime import datetime 


def vae_predict(n_predictions):
    ''' 
    Makes "n" predictions of new watch images. 
    Returns image array of size (n,128,128,3).'''
    INPUT_DIM = (128,128,3)
    # - - - SAME ARCHITECTURE USED IN TRAINING 
    vae = VariationalAutoencoder( 
                    input_dim = INPUT_DIM
                    , encoder_conv_filters=[32,64,64,64,128,128]
                    , encoder_conv_kernel_size=[3,3,3,3,3, 3]
                    , encoder_conv_strides=[2,2,2,2,2, 2]
                    , decoder_conv_t_filters=[64,64,32,3,3, 3]
                    , decoder_conv_t_kernel_size=[3,3,3,3, 3, 3]
                    , decoder_conv_t_strides=[2,2,2,2, 2, 2]
                    , z_dim=200
                    , use_batch_norm=True
                    , use_dropout=True)

    vae.load_weights('run/vae/0005_watches/weights/weights.h5')
    znew = np.random.normal(size = (n_predictions,vae.z_dim))
    reconst = vae.decoder.predict(np.array(znew))
    return reconst





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_predictions', '-n', help='number of predictions to make', type=int)
    parser.add_argument('--save', '-s', help='save the image to the image_results directory',default=True)
    args = parser.parse_args()
    print('args: ', args)

    vae_result = vae_predict(args.n_predictions)
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(vae_result[0])
    if args.save:
        num_existing = str(len(os.listdir('../image_results/')))
        name = '../image_results/VAE_prediction_'+num_existing+'.png'
        fig.savefig(name, dpi=125)



