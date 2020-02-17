from bs4 import BeautifulSoup
import os
from os import path
import shutil
import numpy as np



class WatchScrape():
    def __init__(self):
        self.data_dir = '../data/webpages'
  
    def file2html(self, FILENAME):
        '''given a filename, return the BS4 object for the html file'''
        with open(FILENAME, 'rb') as f:
            html = BeautifulSoup(f, 'html.parser')
            f.close()
        return html



    def collect(self):
        referenced_images = []
        # 1. for each html document in the data_dir, 
        # get all references to img src paths that start 
        # with './Amazon', and get rid of urls, and keep the alt text
        for fileobject in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, fileobject)):
                if fileobject != ".DS_Store":
                    print(f'Scraping Document: {fileobject}')
                    html = file2html(os.path.join(self.data_dir, fileobject))
                    imgs = html.findAll('img')
                    src_list = []
                    for img in imgs:
                        try:
                            src = img.attrs['src']
                            alt = img.attrs['alt']
                            if alt == '' or alt == 'Prime':
                                continue
                            if src[0:13] == './Amazon.com_':
                                referenced_images.append((src,alt))
                        except KeyError as e:
                            continue

        self.referenced_images = referenced_images


        def clean(self):
            # Transform referened_images list into array for indexing
            referenced_images = np.array(self.referenced_images)

            # 2. for all directories in data_dir, extract the names of the referenced images only and delete all other files

            for fileobject in os.listdir(self.data_dir):
                full_dir_path = os.path.join(self.data_dir, fileobject)
                if os.path.isdir(full_dir_path):
                    for file in os.listdir(full_dir_path):
                        full_file_path = os.path.join(full_dir_path,file)

                        # REMOVE THE NON JPG (TARGET) IMAGE FILES
                        if file[-4::] != '.jpg':
                            print('Deleting File: ', file)
                            os.remove(full_file_path)

                        # MOVE THE TARGET FILES TO THE HOLISTIC IMAGE DIRECTORY
                        src = full_dir_path #"C:\\Users\\****\\Desktop\\test1\\"
                        dst = '../data/images/'#"C:\\Users\\****\\Desktop\\test2\\"

                        files = [i for i in os.listdir(src) if i in [os.path.basename(j) for j in referenced_images[:,0] ] ]
                        for f in files:
                            try:
                                shutil.move(path.join(src, f), dst)
                            except:
                                print("already moved this file, continuing...")
                                continue 
            self.cleaned=True

if __name__ == "__main__":
    
    WS = WatchScrape()

    WS.collect()
    WS.clean()

