from skimage import img_as_float
from skimage import io
import skimage.util
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def im_resize(FILEPATH,SIZE,show=False,make_copy=False):
    desired_size = SIZE
    im_pth = FILEPATH#"/home/jdhao/test.jpg"

    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # # use thumbnail() or resize() method to resize the input image

    # # thumbnail is a in-place operation

    # # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)

    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding, fill=(255,255,255))

    if show:
        new_im.show()
    if make_copy:
        FILEPATH = FILEPATH.replace('.jpg',' (copy).jpg')

        new_im.save(FILEPATH)
    return new_im

if __name__ == "__main__":
    
    f = "../data/images/Anne Klein Women's AK-1492MPRG Swarovski Crystal Accented Rose Gold-Tone Bracelet Watch.jpg"

    img = im_resize(f,200,show=True,make_copy=True)

