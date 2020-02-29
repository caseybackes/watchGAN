import os
import random
from skimage import io
import matplotlib.pyplot as plt 

def show_n_original(n=4,m=4,figscale=4):
    # Collect image file paths
    DATA_FOLDER = '../data/images'
    impaths = [os.path.join(DATA_FOLDER, x) for x in  os.listdir(DATA_FOLDER)]

    # Shuffle in place
    random.shuffle(impaths)
    
    # Show in subplot figure
    fig = plt.figure(figsize=(figscale,figscale))
    for i in range(n*m):
        # fig.add_subplot(nrows, ncols, num) 
        ax = fig.add_subplot(n, m, i+1)
        ax.imshow(io.imread(impaths[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return ax 
show_n_original(4,4)