from argparse import ArgumentParser
import itertools

import cv2
import numpy as np


def error_diffuse(i, j, diff_error, error_img):
    """
    Diffuses the error for one pixel value.
    Inputs:
        i: The row index.
        j: The column index.
        diff_error: The error to be diffused.
        error_img: The buffer to save the error.

    Return:
        error_img: The buffer of the error.
    """
    h, w = error_img.shape
    diffuse_pattern = {
        (0, 1): 7 / 16,
        (1, -1): 3 / 16,
        (1, 0): 5 / 16,
        (1, 1): 1 / 16,
    }
    for (delta_i, delta_j), weight in diffuse_pattern.items():
        next_i = i + delta_i
        next_j = j + delta_j
        if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w:
            continue
        error_img[next_i][next_j] += weight * diff_error
    return error_img


def create_pattern(share0, share1, secret):
    """
    Creates the pattern for two share images and the secret image.
    Inputs:
        share0: The counts of the share image 1.
        share1: The counts of the share image 2.
        secret: The counts of the secrete image.

    Returns:
        p_share0: The pattern for the share image 1.
        p_share1: The pattern for the share image 2.
    """
    share0_val = np.zeros(9).astype(np.uint8)
    share0_val[:share0] = 255
    share1_val = np.zeros(9).astype(np.uint8)
    share1_val[:secret] = 255
    if share1 > secret:
        share1_val[secret - share1:] = 255
    indices = np.arange(9)
    np.random.shuffle(indices)
    indices = indices.reshape(3, 3)
    return share0_val[indices], share1_val[indices]


candidates = np.array(
    [[i, j, k]
     for i, j, k in itertools.product(range(10), range(10), range(10))
     if k >= i + j - 9 and k <= min(i, j)]
)


def quantized_points(triplet, candidates=candidates):
    """
    Quantized the triplet of the count under the constraints based on EVC.
    Inputs:
        triplet: The triplet for the values.
        candidates: The candidates under the constraints.

    Returns:
        The closet candidate.
    """
    dist = np.abs(candidates - triplet).sum(axis=-1)
    return candidates[dist.argmin()]


def enc(carry0, carry1, secret):
    """
    Encrypts the secret image from two share images by Extended Visual Cryptography.
    Inputs:
        carry0: The carry images 1.
        carry1: The carry images 2.
        secret: The serect image

    Returns:
        share0, share1: Two share images.
    """
    h, w = secret.shape
    # Dynamic range.

    # Error diffusion.
    error_s0 = np.zeros_like(carry0)
    error_s1 = np.zeros_like(carry1)
    error_s = np.zeros_like(secret)
    share0 = np.zeros((3 * h, 3 * w)).astype(np.uint8)
    share1 = np.zeros((3 * h, 3 * w)).astype(np.uint8)

    bins = np.linspace(0, 255, num=11)
    bins[0] = -np.inf
    bins[-1] = np.inf

    for i in range(h):
        for j in range(w):
            # Quantizes the value.
            count_s0 = np.digitize(carry0[i][j] + error_s0[i][j], bins) - 1
            count_s1 = np.digitize(carry1[i][j] + error_s1[i][j], bins) - 1
            count_s = np.digitize(secret[i][j] + error_s[i][j], bins) - 1

            # Fix the value for the contraints.
            count_s0, count_s1, count_s = quantized_points(
                triplet=np.array([count_s0, count_s1, count_s])
            )
            val_s0 = count_s0 * 255 / 9
            val_s1 = count_s1 * 255 / 9
            val_s = count_s * 255 / 9

            # Diffuses the errors.
            error_diffuse(i, j,
                          carry0[i][j] + error_s0[i][j] - val_s0, error_s0)
            error_diffuse(i, j,
                          carry1[i][j] + error_s1[i][j] - val_s1, error_s1)
            error_diffuse(i, j,
                          secret[i][j] + error_s[i][j] - val_s, error_s)

            # Assigns the patterns.
            p_share0, p_share1 = create_pattern(count_s0, count_s1, count_s)
            share0[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = p_share0
            share1[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = p_share1
    return share0, share1


def dec(share0: np.array, share1: np.array):
    """
    Decrypts the secret image from two share images by Extended Visual Cryptography.
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

        share0, share1 = enc(carry0=carry0, carry1=carry1, secret=secret)

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

    main(parser.parse_args())
