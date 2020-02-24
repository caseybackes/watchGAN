import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import pandas as pd

from VAE_MODEL import VariationalAutoencoder
from utils.loaders import load_model, ImageLabelLoader

# run params
section = 'vae'
run_id = '0003'
data_name = 'watches'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])
IMAGE_FOLDER = '../data/train/processed_images/'



INPUT_DIM = (128,128,3)



#imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])


#vae = load_model(VariationalAutoencoder, RUN_FOLDER)
vae = VariationalAutoencoder(
                input_dim = INPUT_DIM
                , encoder_conv_filters=[32,64,64, 64]
                , encoder_conv_kernel_size=[3,3,3,3]
                , encoder_conv_strides=[2,2,2,2]

                , decoder_conv_t_filters=[64,64,32,3]
                , decoder_conv_t_kernel_size=[3,3,3,3]
                , decoder_conv_t_strides=[2,2,2,2]
                , z_dim=200
                , use_batch_norm=True
                , use_dropout=True)


n_to_show = 30

znew = np.random.normal(size = (n_to_show,vae.z_dim))

reconst = vae.decoder.predict(np.array(znew))

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(3, 10, i+1)
    ax.imshow(reconst[i, :,:,:])
    ax.axis('off')

name = 'predictions-' + run_id+'-watches.png'
fig.savefig(name, dpi=125)
