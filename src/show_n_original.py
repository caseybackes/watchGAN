import os
import random
from skimage import io
import matplotlib.pyplot as plt 
import argparse


def show_n_original(n_rows=4,m_cols=4,plotscale=4, save_plot=False):
    """Displays a random subset of original images of watches and returns a 
    figure. 

    Parameters
    ----------
    n_rows : int
        number of grid rows to display
    
    m_cols : int
        number of grid cols to display

    plotscale : int
        scaling factor for the size of the figure

    Returns
    -------
    matplotlib.pyplot.figure
        The figure object of the resulting display of images
    """

    # Collect image file paths
    DATA_FOLDER = '../data/images'
    impaths = [os.path.join(DATA_FOLDER, x) for x in  os.listdir(DATA_FOLDER)]

    # Shuffle in place
    random.shuffle(impaths)
    
    # Show in subplot figure
    fig = plt.figure(figsize=(plotscale,plotscale))
    for i in range(n_rows*m_cols):
        # fig.add_subplot(nrows, ncols, num) 
        ax = fig.add_subplot(n_rows, m_cols, i+1)
        ax.imshow(io.imread(impaths[i]))
        ax.axis('off')
    plt.tight_layout()
    
    # save the plot if requested and return figure
    if save_plot:
        how_many= len(os.listdir('../image_results'))
        plt.savefig('../image_results/original_grid-'+ str(how_many)+'.png', transparent=True)
    plt.show()
    return fig 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce grid of (n x m) original images.')
    parser.add_argument('--n_rows','-n', type=int, 
                            help='number of rows in grid plot',
                            default=4)
    parser.add_argument('--m_cols','-m', type = int,
                            help='number of cols in grid plot',
                            default=4)

    parser.add_argument('--plotscale','-f', type=int,
                            help='scale size of plot when displaying results')
    parser.add_argument('--save','-s', action='store_true', dest='save', 
                            help='opt flag to save the prediction plot image')
    args = parser.parse_args()
    
    show_n_original(4,4)
    show_n_original(args.n_rows, args.m_cols, args.plotscale, args.save)