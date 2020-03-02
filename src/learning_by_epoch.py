import matplotlib.pyplot as plt 
import os
import numpy as np
from skimage import io 



def show_learning(vae_model_id=16):
    # Retrieve the first image prediction for every ten epochs of training
    run_dir = 'run/vae/0000'
    run_folder = run_dir[0:-len(str(vae_model_id))]+str(vae_model_id)+'_watches/images/'  # 'run/vae/0016'  
    
    # Get image file names from run folder
    img_list = os.listdir(run_folder)
    imgformat = 'img_{}_0.jpg'
    plot_scale = 1.5
    fig = plt.figure(figsize=(plot_scale*5, 8*plot_scale),dpi=120)
    epoch_id_list = [int(x) for x in np.linspace(0,400,40)]
    for i in np.linspace(1,40,40):
        # format the image name to read/plot
        epoch_id = str(int(i*10)).zfill(3) # '010'
        img_name = run_folder +  imgformat.format(epoch_id) # 'img_010_0.jpg'

        # retrieve image
        imgdata = io.imread(img_name)
        
        # make a new axis to plot this image on 
        ax = fig.add_subplot(5,8, i)  
        ax.axis('off')
        ax.text(x=.5,y=-.5,s=str(epoch_id))
        ax.imshow(imgdata)
    fig.tight_layout(pad=1)
    plt.show()
    # make a plot

    # save the plot image under 'image_results'


if __name__ == "__main__":
    show_learning()