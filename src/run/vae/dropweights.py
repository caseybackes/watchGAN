import os
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="number of run to drop interim weights from within the n_watches/ weights dir")
    args = parser.parse_args()

    print('Dropping interim weights for run ', args.n,'_watches...')

    for w in os.listdir(args.n+ '_watches/weights'):
        if w != 'weights.h5':
            pth = os.path.join(args.n+'_watches/weights')
            os.remove(os.path.join(pth,w))

    print('OS.LISTDIR\n',os.listdir(args.n+'_watches/weights'))
    #0004_watches  0003_watches ...
    #weights_dir = 
    #0004_watches/weights/

