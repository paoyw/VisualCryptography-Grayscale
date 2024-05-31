import numpy as np
import cv2
from pathlib import Path
import copy
import time

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = 'my description')
    parser.add_argument('--secret_img', type = str)
    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    src_img = cv2.imread(args.secret_img + '.png', cv2.IMREAD_GRAYSCALE)
    _src_img = np.where( src_img < 127, 1, 0 )

    height, width = src_img.shape

    rng = np.random.default_rng()

    block_size = (2, 2)

    start = time.time()

    # Here we only apply 2 shares to finish the baseline.
    share1, share2 = \
        np.zeros( (height * block_size[0], width * block_size[1]), dtype = 'uint8' ), \
        np.zeros( (height * block_size[0], width * block_size[1]), dtype = 'uint8' )

    black_pixels = np.prod( block_size )
    white_pixels = black_pixels // 2

    # Create share image.
    for i in range(height):
        for j in range(width):
            block1 = np.zeros( block_size )
            block2 = np.zeros( block_size )
            # Black.
            if _src_img[i, j] == 1:
                print('black.')
                index_list = rng.permutation(black_pixels)
                block1.ravel()[index_list[: black_pixels // 2]] = 1
                block2.ravel()[index_list[black_pixels // 2 :]] = 1
            # White.
            elif _src_img[i, j] == 0:
                print('white.')
                index_list = rng.permutation(black_pixels)
                block1.ravel()[index_list[: black_pixels // 2]] = 1
                block2.ravel()[index_list[: black_pixels // 2]] = 1
                # block2.ravel()[: black_pixels // 2] = 1
            share1[ i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1] ] = block1
            share2[ i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1] ] = block2

    print(_src_img)
    print(share1.dtype)
    print(share2.dtype)

    end = time.time()
    print(f'Used_time: {end - start}')

    # Output share.
    cv2.imwrite(f'share_1.png', 255 - share1 * 255)
    cv2.imwrite(f'share_2.png', 255 - share2 * 255)

    # Do the stacking.
    stacked_img = share1 | share2
    resized_img = cv2.resize(255 - stacked_img * 255, (192, 192))
    print(stacked_img)
    cv2.imwrite(f'stacked.png', 255 - stacked_img * 255)
    cv2.imwrite(f'resized.png', resized_img)


