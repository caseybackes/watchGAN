from bs4 import BeautifulSoup
import os
from os import path
import shutil
import numpy as np
import pandas as pd


class WatchScrape():
    def __init__(self):
        self.data_dir = '../data/webpages/'
        self.image_dir = '../data/images/'

    def collect(self):
        for f in os.listdir(self.data_dir):
            fpath = os.path.join(self.data_dir,f)
            if os.path.isfile(fpath):
                with open(fpath,'r') as fopen:
                    html = BeautifulSoup(fopen, 'html.parser')
                    self.html = html
                    fopen.close()
                imgs = self.html.findAll('img', attrs={'class':'s-image'})
                for img in imgs:
                    src = img.attrs['src'].lstrip('./')
                    alt = img.attrs['alt'].replace('/','-')
                    try:
                        shutil.move(f'{self.data_dir}{src}', f"{self.image_dir}{alt}.jpg")
                    except FileNotFoundError as fnf:
                        print(img, ' moved already')
        for f in os.listdir(self.data_dir):
            fpath = os.path.join(self.data_dir,f)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)

ws = WatchScrape()
ws.collect()

    # - - - Extract the names of the relevant images from the data/webpages2 directory
    
    # - - - Move the relevant file to the data/images2 directory with rename to alt_txt

    # - - - Delete the irrelevant data directory and keep the html file

