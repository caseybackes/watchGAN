from vae_predict import vae_predict
import os
from glob import glob
import numpy as np
from vae.VAE_MODEL import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator

DATA_FOLDER = '../data/train' #contains the class dir of 'processed_images'

