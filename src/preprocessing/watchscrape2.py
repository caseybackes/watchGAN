from bs4 import BeautifulSoup
import os
from os import path
import shutil
import numpy as np
import pandas as pd


class WatchScrape():
    def __init__(self):
        self.data_dir = '../../data/webpages/'
        self.image_dir = '../../data/images/'

    def collect(self):
        for f in os.listdir(self.data_dir):
            print('f')
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
                        continue
                        #print(img, ' moved already')
        for f in os.listdir(self.data_dir):
            fpath = os.path.join(self.data_dir,f)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)

ws = WatchScrape()
ws.collect()

