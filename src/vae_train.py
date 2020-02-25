import os
from glob import glob
import numpy as np

from vae.VAE_MODEL import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator





# ITERATION PARAMETERS
section = 'vae'
run_id = '0005'
data_name = 'watches'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])




if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #


DATA_FOLDER = '../data/train' #contains the class dir of 'processed_images'


INPUT_DIM = (128,128,3)
BATCH_SIZE = 32

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))

NUM_IMAGES = len(filenames)

noise_maker = lambda x: x + np.random.uniform(size=(INPUT_DIM))

data_gen = ImageDataGenerator(rescale=1./255
                                , rotation_range=4
                                , width_shift_range=.1
                                , height_shift_range=.1
                                , horizontal_flip=True,
                                preprocessing_function= noise_maker)

data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                         , target_size = INPUT_DIM[:2]
                                         , batch_size = BATCH_SIZE
                                         , shuffle = True
                                         , classes=None
                                         , class_mode = 'input'
                                         , subset = "training"
                                            )



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

if mode == 'build':
    vae.save('../data/savedmodels/')
    vae.save('run/vae/'+str(run_id + '_watches'))
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# vae.encoder.summary()
# vae.decoder.summary()


LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 2000#5000#10000
EPOCHS = 400
PRINT_EVERY_N_BATCHES = 10
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

vae.train_with_generator(     
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)
