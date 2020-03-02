from PIL import Image
import os
import numpy as np




def make_gif(run_id=16):

    dirname = 'run/vae/0000'[0:-len(str(run_id))]+str(run_id) + '_watches/images/'
    img_names = sorted(os.listdir(dirname))
    num_imgs = len(img_names)

    images = []

    for num,img in enumerate(img_names):

        img_data = np.array(Image.open(dirname+img))
        percent_done = num/len(img_names)

        img_data[-5::,0:int((num/len(img_names))*img_data.shape[1])] = np.array([0,255,0])  
        print('Progress: ',int((num/len(img_names))*img_data.shape[1]) )
        img_data = Image.fromarray(img_data)

        images.append(img_data)
        # print('adding ',img)


    images[0].save('testwatch.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

make_gif()