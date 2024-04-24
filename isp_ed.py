from argparse import ArgumentParser

import cv2
import numpy as np

def main(args):
    secret_img = cv2.imread(args.secret, cv2.IMREAD_GRAYSCALE).astype(np.float32)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--secret')
    parser.add_argument('--output')

    main(parser.parse_args())