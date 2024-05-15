from argparse import ArgumentParser
import random

import cv2
import numpy as np

from utils.halftone import errorDiffusionFloydSteinberg


W = {
    (-2, -2): 1,
    (-2, -1): 2,
    (-2, 0): 2,
    (-2, 1): 2,
    (-2, 2): 1,
    (-1, -2): 2,
    (-1, -1): 4,
    (-1, 0): 5,
    (-1, 1): 4,
    (-1, 2): 2,
    (0, -2): 2,
    (0, -1): 5,
    (0, 0): 6,
}

J = {
    (-2, -2): 1,
    (-2, -1): 3,
    (-2, 0): 5,
    (-2, 1): 3,
    (-2, 2): 1,
    (-1, -2): 3,
    (-1, -1): 5,
    (-1, 0): 7,
    (-1, 1): 5,
    (-1, 2): 3,
    (0, -2): 5,
    (0, -1): 7,
}


def compute_density(i, j, H0, H1, h, w):
    val_sum = 0
    w_sum = 0
    for (_i, _j), weight in W.items():
        _i += i
        _j += j
        if _i >= 0 and _i < h and _j >= 0 and _j < w:
            val_sum += int(H0[_i][_j]) & int(H1[_i][_j])
            w_sum += weight
    return val_sum / w_sum


def compute_T(i, j, h, w):
    w_sum = 0
    for (_i, _j), weight in W.items():
        _i += i
        _j += j
        if _i >= 0 and _i < h and _j >= 0 and _j < w:
            w_sum += weight
    return 6 / w_sum / 2


compute_Tw = compute_T
compute_Tb = compute_T


def accum_error(i, j, error, h, w):
    w_sum = 0
    error_sum = 0
    for (_i, _j), weight in J.items():
        _i += i
        _j += j
        if _i >= 0 and _i < h and _j >= 0 and _j < w:
            error_sum += error[_i][_j] * weight
            w_sum += weight
    if w_sum == 0:
        return 0
    return error_sum / w_sum


def dynamic_range_control(img, threshold0=0.25, threshold1=0.75):
    "Classify the pixel value to three class by thresholds."
    if not threshold0:
        threshold0 = np.percentile(img, 0.25)
    if not threshold1:
        threshold1 = np.percentile(img, 0.75)
    result = 0.5 * np.ones_like(img, dtype=np.float32)
    result = np.where(img < threshold0, 0, result)
    result = np.where(img >= threshold1, 1, result)
    return result


def expand(h):
    _h, _w = h.shape
    H = np.zeros((2 * _h, 2 * _w))
    for i in range(_h):
        for j in range(_w):
            if h[i][j] > 0:
                H[2 * i:2 * i + 2, 2 * j: 2 * j + 2] = random.choice([
                    np.array([[1, 0], [0, 1]]),
                    np.array([[0, 1], [1, 0]]),
                ])
    return H


def enc(carry0, carry1, secret, threshold=0.5, delta=0.05):
    """
    Encrypts the secret image from two share images.
    Inputs:
        carry0: The carry images 1.
        carry1: The carry images 2.
        secret: The serect image

    Returns:
        share0, share1: Two share images.
    """
    h, w = carry0.shape

    carry0 = 0.45 * carry0 / 255 + 0.275
    carry1 = 0.45 * carry1 / 255 + 0.275
    secret = 0.45 * secret / 255

    h0 = errorDiffusionFloydSteinberg(carry0, threshold=0.5, value=1)
    H0 = expand(h0)

    h1 = np.zeros_like(h0)
    H1 = np.zeros_like(H0)
    error1 = np.zeros_like(h0)

    for i in range(h):
        for j in range(w):
            if carry1[i][j] + accum_error(i, j, error1, h, w) > threshold:
                h1[i][j] = 1
            else:
                h1[i][j] = 0

            if np.abs(carry1[i][j] + accum_error(i, j, error1, h, w) - threshold) < delta:
                d = compute_density(i, j, h0, h1, h, w)
                if (secret[i][j] - d > compute_Tb(i, j, h, w)):
                    H1[2 * i:2 * i + 2, 2 * j: 2 * j + 2] = \
                        H0[2 * i:2 * i + 2, 2 * j: 2 * j + 2]
                elif (d - secret[i][j] > compute_Tw(i, j, h, w)):
                    H1[2 * i:2 * i + 2, 2 * j: 2 * j + 2] = \
                        H0[2 * i:2 * i + 2, 2 * j: 2 * j + 2].astype(np.uint8) ^ 1
                elif carry1[i][j] + accum_error(i, j, error1, h, w) > threshold:
                    H1[2 * i:2 * i + 2, 2 * j: 2 * j + 2] = random.choice([
                        np.array([[1, 0], [0, 1]]),
                        np.array([[0, 1], [1, 0]]),
                    ])
            elif carry1[i][j] + accum_error(i, j, error1, h, w) > threshold:
                H1[2 * i:2 * i + 2, 2 * j: 2 * j + 2] = random.choice([
                    np.array([[1, 0], [0, 1]]),
                    np.array([[0, 1], [1, 0]]),
                ])

            error1[i][j] = carry1[i][j] - h1[i][j]
    return 255 * H0.astype(np.uint8), 255 * H1.astype(np.uint8)


def dec(share0, share1):
    """
    Decrypts the secret image from two share images.
    Inputs:
        share0: The share images 1.
        share1: The share images 2.

    Returns:
        output: The decrypt result.
    """
    return np.where(share0 & share1, 255, 0)


def main(args):
    if not args.dec_only:
        carry0 = cv2.imread(
            args.carry0, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        carry1 = cv2.imread(
            args.carry1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        secret = cv2.imread(
            args.secret, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        share0, share1 = enc(carry0=carry0, carry1=carry1, secret=secret,
                             threshold=args.threshold, delta=args.delta)

        if args.share0:
            cv2.imwrite(args.share0, share0)
        if args.share1:
            cv2.imwrite(args.share1, share1)
    else:
        share0 = cv2.imread(args.share0, cv2.IMREAD_GRAYSCALE)
        share1 = cv2.imread(args.share1, cv2.IMREAD_GRAYSCALE)

    if args.dec:
        dec_img = dec(share0, share1)
        cv2.imwrite(args.dec, dec_img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--carry0')
    parser.add_argument('--carry1')
    parser.add_argument('--secret')
    parser.add_argument('--share0')
    parser.add_argument('--share1')

    parser.add_argument('--dec')
    parser.add_argument('--dec-only', action='store_true')

    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--delta', default=0.06, type=float)

    main(parser.parse_args())
