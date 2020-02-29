from skimage import img_as_float
from skimage import io
import skimage.util
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os




def im_resize(FILEPATH,SIZE,show=False,make_copy=False):
    desired_size = SIZE
    im_pth = FILEPATH
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)

    # determine pixel change 
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))


    try:
        # reconstructing the image with the new size and whitespace fill. 
        new_im = ImageOps.expand(im, padding, fill=(255,255,255))

    except TypeError as te:
        print(te)
        return None 

    if show:
        new_im.show()
    if make_copy:
        FILEPATH = '../../data/images/size256/processed_images/'+os.path.basename(FILEPATH)
        new_im.save(FILEPATH)
        print('COPIED: ', os.path.basename((FILEPATH)))
    return new_im

if __name__ == "__main__":    
    
    image_dir = '../../data/images/'
    for imgpath in os.listdir(image_dir):
        if imgpath != '.DS_Store':
            try:
                blank = im_resize(image_dir + imgpath,256,show=False,make_copy=True)
            except:
                pass 
    print('Resized and sqared the images were added to the data/process_images directory.')
