import os
import random
from glob import glob
import numpy as np
import argparse 
from vae.VAE_MODEL import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator



def vae_train(image_depth=1000
            , keep_model=False
            , epochs=10
            , mode = 'build'
            , r_loss_factor = 10000):

    # DATA_FOLDER = '../data/train' #contains the class dir of 'processed_images'
    DATA_FOLDER = '../data/images/size256' #contains the class dir of 'processed_images'

    # ITERATION PARAMETERS
    section = 'vae'
    how_many_runs = len(os.listdir('run/vae'))
    # run_id = '0006'
    run_id = str('0000000'+str(how_many_runs))[-4:]
    data_name = 'watches256'
    RUN_FOLDER = 'run/{}/'.format(section)
    RUN_FOLDER += '_'.join([run_id, data_name])

    # Save the params for this run
    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    # mode =  'build' #'load' #


    INPUT_DIM = (256,256,3)
    BATCH_SIZE = 32

    # Get image file names and shuffle
    filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
    random.shuffle(filenames)
    filenames = filenames[0:image_depth]#None 
    NUM_IMAGES = len(filenames)

    
    # Feed data to the model iteratively instead of holding all in memory
    data_gen = ImageDataGenerator(rescale=1./255)
    data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                            , target_size = INPUT_DIM[:2]
                                            , batch_size = BATCH_SIZE
                                            , shuffle = True
                                            , classes=None
                                            , class_mode = 'input'
                                            , subset = "training")

    # Define the VAE model
    vae = VariationalAutoencoder(
                    input_dim = INPUT_DIM
                    , encoder_conv_filters=[32,64,64,64,128,128,128,128]
                    , encoder_conv_kernel_size=[3,3,3,3,3,3,3,3]
                    , encoder_conv_strides=[2,2,2,2,2,2,2,2]
                    , decoder_conv_t_filters=[128,128,64,64,32,3,3, 3]
                    , decoder_conv_t_kernel_size=[3,3,3,3,3,3, 3, 3]
                    , decoder_conv_t_strides=[2,2,2,2,2,2, 2, 2]
                    , z_dim=200
                    , use_batch_norm=True
                    , use_dropout=True)
    
    # Save the model architecture
    if mode == 'build':
        # vae.save('../data/savedmodels/')
        vae.save('run/vae/'+str(run_id + '_watches256'))
    else:
        vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    # vae.encoder.summary()
    # vae.decoder.summary()

    # Train the model
    LEARNING_RATE = 0.0005
    R_LOSS_FACTOR = r_loss_factor#10000
    EPOCHS = epochs 
    PRINT_EVERY_N_BATCHES = 10
    INITIAL_EPOCH = 0

    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)
    print('*'*20,'\nTRAINING THE VAE MODEL NOW...')

    vae.train_with_generator(     
        data_flow
        , epochs = EPOCHS
        , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
        , run_folder = RUN_FOLDER
        , print_every_n_batches = PRINT_EVERY_N_BATCHES
        , initial_epoch = INITIAL_EPOCH
    )

    if keep_model:
        how_many = sum(['vae' in x for x in os.listdir('run/vae')])+1
        # name = 'run/vae/vae_model'+str(how_many+1)+'.h5'
        name= RUN_FOLDER + '/vae.h5'
        vae.save(name)
        print("Saved autoencoder model under ", name)
    
    return vae 


if __name__ == "__main__":
    

    # ARGPARSE FOR HYPERPERAMETERS
    parser = argparse.ArgumentParser(description='Train a variational autoencoder model.')
    parser.add_argument('--imdepth','-d', type=int, help='number of images to train model on',default=None)
    parser.add_argument('--epochs','-e', type = int,help='number of epochs for training',required=True)
    parser.add_argument('--rls','-r', type = int,help='r loss factor for vae model,defaults to 10,000',default=10000)
    parser.add_argument('--save','-s', action='store_true', dest='save', help='opt to save the model')
    parser.add_argument('--mode','-m', type=str, help='build or load', default='build')
    args = parser.parse_args()

    # The case of using all images to train VAE model
    if not args.imdepth:
        args.imdepth=-1
    # breakpoint()
    # Generate the VAE model  
    vae_train(image_depth=args.imdepth
            , keep_model=args.save
            , epochs=args.epochs
            , mode = args.mode
            , r_loss_factor=args.rls)