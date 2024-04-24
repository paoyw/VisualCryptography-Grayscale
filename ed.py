from argparse import ArgumentParser

import cv2
import numpy as np


def main(args):
    s0 = cv2.imread(args.share0, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s1 = cv2.imread(args.share1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    secret_img = cv2.imread(
        args.secret, cv2.IMREAD_GRAYSCALE).astype(np.float32)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--share0')
    parser.add_argument('--share1')
    parser.add_argument('--secret')
    parser.add_argument('--output')

    main(parser.parse_args())
