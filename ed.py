from argparse import ArgumentParser

import cv2
import numpy as np


def main(args):
    carry0 = cv2.imread(args.carry0, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    carry1 = cv2.imread(args.carry1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    secret = cv2.imread(args.secret, cv2.IMREAD_GRAYSCALE).astype(np.float32)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--carry0')
    parser.add_argument('--carry1')
    parser.add_argument('--secret')
    parser.add_argument('--share0')
    parser.add_argument('--share1')

    main(parser.parse_args())
