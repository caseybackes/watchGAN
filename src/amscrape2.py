from bs4 import BeautifulSoup
import os


data_dir = '../data/webpages'

def file2html(FILENAME):
    '''given a filename, return the BS4 object for the html file'''
    with open(FILENAME, 'rb') as f:
        html = BeautifulSoup(f, 'html.parser')
        f.close()
    return html


# html_list = []
# all_image_paths = []

# for f in os.listdir(data_dir):
#     # fully qualified absolute path to fileobject
#     full_f = os.path.join(data_dir, f)
#     print('full f: ', full_f)

#     # if dir then get all image names from dir
#     if os.path.isdir(full_f):
#         for file in os.listdir(full_f):
#             if '.jpg' in os.path.basename(file):
#                 all_image_paths.append(os.path.join(full_f,file))
#     # if html file, then transform to bs4 obj
#     else:
#         html_list.append(
#             file2html(full_f)
#             )



data_dir = '../data/webpages'
# files = os.listdir(data_dir)
referenced_images = []
alt_text_list = []
# 1. for each html document in the data_dir, get all references to img src paths that start with './Amazon', and get rid of urls, and keep the alt text
for fileobject in os.listdir(data_dir)[0:4]:
    if os.path.isfile(os.path.join(data_dir, fileobject)):
        if fileobject != ".DS_Store":
            print(f'{fileobject} is a document')
            html = file2html(os.path.join(data_dir, fileobject))
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

       

# for all directories in data_dir, extract the names of the referenced images only and delete all other files

for fileobject in os.listdir(data_dir)[0:-1]:
    full_dir_path = os.path.join(data_dir, fileobject)
    if os.path.isdir(full_dir_path):
        for file in os.listdir(full_dir_path):
            full_file_path = os.path.join(full_dir_path,file)
            if file[-4::] != '.jpg':
                print('I want to delete ', file)
                os.remove(full_file_path)








# html = BeautifulSoup(open('/Users/casey/Documents/DataScienceProjects/watchgan/data/webpages/Amazon.com_ watch.html'), 'html.parser')
# img100 = html.find_all('img')[100]
# img100.attrs['src'] 


