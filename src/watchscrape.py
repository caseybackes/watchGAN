from bs4 import BeautifulSoup
import os
from os import path
import shutil
import numpy as np



class WatchScrape():
    def __init__(self):
        self.data_dir = '../data/webpages'
        self.is_culled = False
        self.referenced_images=None
        self.html = []
  
    def file2html(self, FILENAME):
        '''given a filename, return the BS4 object for the html file'''
        with open(FILENAME, 'rb') as f:
            html = BeautifulSoup(f, 'html.parser')
            f.close()
        return html

    def collect(self):
        ''' These are the image src strings that are likely to be associated with
        watch images in the webpage files'''
        referenced_images = []
        # 1. for each html document in the data_dir, 
        # get all references to img src paths that start 
        # with './Amazon', and get rid of urls, and keep the alt text
        for fileobject in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, fileobject)):
                if fileobject != ".DS_Store":
                    print(f'Scraping Document: {fileobject}')
                    html = self.file2html(os.path.join(self.data_dir, fileobject))
                    self.html.append(html)
                    imgs = html.findAll('img',attrs={'class':'s-image'})
                    src_list = []
                    for img in imgs:
                        try:
                            src = img.attrs['src']
                            alt = img.attrs['alt']
                            if alt == '' or alt == 'Prime':
                                continue
                            if 'amazon' in src or 'prime' in src or 'fashion' in src:
                                continue

                            if src.startswith('./Amazon.com_'):
                                referenced_images.append((src,alt))
                        except KeyError as e:
                            continue
        self.referenced_images = np.array(referenced_images)

    def cull(self):
        ''' 
        This method deletes the non-jpg files in the 'webpages' directory. Then moves the 
        appropriate image files to the aggregate 'images' directory. Deletes the files_i directory. 

        '''
        # Transform referened_images list into array for indexing
        referenced_images = self.referenced_images
        # 2. for all directories in data_dir, extract the names of the referenced images only and delete all other files
        for fileobject in os.listdir(self.data_dir): # for each fileobj/dir in data_dir
            full_dir_path = os.path.join(self.data_dir, fileobject) # fully qualified path to fileobj/dir
            if os.path.isdir(full_dir_path): # if fileobj/dir is dir:
                for file in os.listdir(full_dir_path): # for each file in dir
                    full_file_path = os.path.join(full_dir_path,file) # flly qualifed path to file

                    # REMOVE THE NON-JPG IMAGE FILES
                    if file[-4::] != '.jpg':
                        print('Deleting File: ', file)
                        try:
                            os.remove(full_file_path)
                        except FileNotFoundError as e:
                            continue # whatever. At least we tried to get rid of it. 

                    # MOVE THE TARGET FILES IN THIS DIRTO THE AGGREGATE IMAGE DIRECTORY
                    src = full_dir_path #"C:\\Users\\****\\Desktop\\test1\\"
                    dst = '../data/images/'#"C:\\Users\\****\\Desktop\\test2\\"

                    files = [i for i in os.listdir(src) if i in [os.path.basename(j) for j in referenced_images[:,0] ] ]
                    for f in files:
                        try:
                            shutil.move(path.join(src, f), dst)
                        except:
                            print(f"Failed to move {f}")
                            continue 

                # DANGER ZONE!!
                #shutil.rmtree(full_dir_path)   # effectively: os.remove(full_dir_path)

        self.is_culled=True

if __name__ == "__main__":
    
    WS = WatchScrape()

    WS.collect()
    WS.cull()

