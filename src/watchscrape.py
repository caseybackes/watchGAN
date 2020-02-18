from bs4 import BeautifulSoup
import os
from os import path
import shutil
import numpy as np
import pandas as pd



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
                if fileobject != ".DS_Store" and '.txt' not in fileobject:
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
                    try:
                        os.remove(os.path.join(self.data_dir, fileobject))
                    except FileNotFoundError as ef:
                        print(ef)
                        continue
        self.referenced_images = np.array(referenced_images)

    def write_refimg(self):
        ref_file = '../data/webpages/reflist.txt'
        with open(ref_file, 'a') as f:
            for pair in self.referenced_images:
                f.write(f'{os.path.basename(pair[0])}\t{pair[1]}\n')
            f.close()

    def cull(self):
        ''' 
        This method deletes the non-jpg files in the 'webpages' directory. Then moves the 
        appropriate image files to the aggregate 'images' directory. Deletes the files_i directory. 

        '''
        # Transform referened_images list into array for indexing
        referenced_images = self.referenced_images
        # 2. for all directories in data_dir, extract the names of the referenced images only and delete all other files
        for fileobject in os.listdir(self.data_dir): # for each fileobj/dir in 'webpages/'
            full_dir_path = os.path.join(self.data_dir, fileobject) # fully qualified path to each item in 'webpages'/
            if os.path.isdir(full_dir_path): # if fileobj/dir is dir:
                for file in os.listdir(full_dir_path): # for each file in dir
                    full_file_path = os.path.join(full_dir_path,file) # flly qualifed path 

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
                            #print(f"Failed to move {f}")
                            continue 

                # DANGER ZONE!!
                shutil.rmtree(full_dir_path)   # effectively: os.remove(full_dir_path) because we got everything we wanted from these files

        self.is_culled=True

    def to_df(self):
        reflist = open('../data/webpages/reflist.txt', 'r').readlines()
        refs = []

        for line in reflist:
            split_line = line.split('\t')
            if len(split_line)>2:
                split_line_a = split_line[0]
                split_line_b = ' '.join(split_line[1:])
                split_line = [split_line_a,split_line_b]
                print("Fixed split line: ", split_line)
            a,b = split_line
            b = b.rstrip('\n')
            refs.append([a,b])
        refs = np.array(refs)
        df = pd.DataFrame(columns = refs[0], data = refs[1::]).drop_duplicates()
        df.reset_index().drop('index',axis=1,inplace=True)
        df.to_csv(r'../data/webpages/reflist.txt', header=None, index=None, sep='\t', mode='a') 
        self.df = df

    def reconcile_reflist(self):
        ''' Sometimes an images gets in and upon visual inspection, it needs to be removed (ie: not an image of a watch).
        When deleted from the directory via Finder (in MacOS), that image file will not longer be relevent in the reflist.txt
        file. So we should remove it from the text file. 
        '''
        with open('../data/webpages/reflist.txt', 'r') as f:
            reflist= f.readlines()
            f.close()

        lines_to_remove = []
        for num,line in enumerate(reflist[1::]): # not including the header, obviously. 
            fname = line.split('\t')[0] 
            if fname not in os.listdir('../data/images'): 
                print(fname, ' does not exist in data/images') 
                reflist.remove(line)
                #lines_to_remove.append(num)
        with open('../data/webpages/reflist.txt', 'w') as f:
            # f.write('filename\talt_text\n')
            for line in reflist:
                f.write(line)
            f.close()

        self.referenced_images= reflist



if __name__ == "__main__":
    
    WS = WatchScrape()

    WS.collect()
    WS.write_refimg()
    WS.reconcile_reflist()
    WS.cull()
    WS.to_df()

    WS.df['alt_text'] = WS.df['alt_text'].apply(lambda x: x.replace('"', ''))
    WS.df.drop_duplicates(inplace=True)
    WS.df.reset_index(inplace=True)
    WS.df.drop('index', axis=1, inplace=True)
    os.remove('../data')
    WS.df.to_csv(r'../data/webpages/reflist.txt', header='filename\talt_text\n', index=None, sep='\t', mode = 'a') 








