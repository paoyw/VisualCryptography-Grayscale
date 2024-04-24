from argparse import ArgumentParser

import cv2
import numpy as np


def isp_ed_enc(secret_img: np.array,
               alpha: int = 1, gamma: float = 0.5,
               threshold: float = 0.5):
    """
    Encrypt the secret image into two share images by Image Size-Preserving Visual Cryptography by Error Diﬀusion.

    Inputs:
        secret_img: The secret image.
        alpha: The scaling factor for normalization.
        gamma: The initialize value for the carry images.
        threshold: The threshold for error diffusion for share images.

    Returns:
        share_imgs: Two share images.
    """
    h, w = secret_img.shape
    secret_img = secret_img.astype(np.float32)

    # Normalized the image
    secret_img = alpha * \
        (secret_img - secret_img.min()) / \
        (secret_img.max() - secret_img.min())

    share0 = np.zeros_like(secret_img)
    share1 = np.zeros_like(secret_img)

    error_img = np.zeros_like(secret_img)
    error0 = np.zeros_like(secret_img)
    error1 = np.zeros_like(secret_img)

    carry0 = gamma * np.ones_like(secret_img)
    carry1 = gamma * np.ones_like(secret_img)

    b_patterns = [np.array([[0], [0]]),
                  np.array([[1], [0]]),
                  np.array([[0], [1]])]

    weights = np.array([
        [0, 0, 7 / 16],
        [3 / 16, 5 / 16, 1 / 16]
    ])

    for i in range(h):
        for j in range(w):
            if secret_img[i][j] + error_img[i][j] > threshold:
                b = 1
                d = secret_img[i][j] + error_img[i][j] - b
                share0[i][j] = 1
                share1[i][j] = 1
                delta = np.array([[carry0[i][j] + error0[i][j]],
                                  [carry1[i][j] + error1[i][j]]]) - 1
            else:
                b = 0
                d = secret_img[i][j] + error_img[i][j] - b
                c_e = np.array([[carry0[i][j] + error0[i][j]],
                                [carry1[i][j] + error1[i][j]]])
                t = 0
                min_val = np.inf
                for k, b_pattern in enumerate(b_patterns):
                    val = np.linalg.norm(c_e - b_pattern, ord=1)
                    if val < min_val:
                        t = k
                        min_val = val

                share0[i][j] = b_patterns[t][0][0]
                share1[i][j] = b_patterns[t][1][0]

                delta = c_e - b_patterns[t]

            # Error diffusion.
            for k, l in ((0, 1), (1, -1), (1, 0), (1, 1)):
                next_i = i + k
                next_j = j + l
                if next_i >= 0 and next_i < h \
                        and next_j >= 0 and next_j < w:
                    error_img[next_i][next_j] += weights[k][l + 1] * d

                    error0[next_i][next_j] += weights[k][l + 1] * delta[0][0]
                    error1[next_i][next_j] += weights[k][l + 1] * delta[1][0]

    share0 = (255 * share0).clip(0, 255).astype(np.uint8)
    share1 = (255 * share1).clip(0, 255).astype(np.uint8)
    return share0, share1


def isp_ed_dec(share0: np.array, share1: np.array):
    """
    Dencrypt the secret image from two share images by Image Size-Preserving Visual Cryptography by Error Diﬀusion.
    Inputs:
        share0: The share images 1.
        share1: The share images 2.
    
    Returns:
        output: The decrypt result.
    """
    return np.where(share0 & share1, 255, 0)


def main(args):
    if not args.dec_only:
        secret_img = cv2.imread(args.secret, cv2.IMREAD_GRAYSCALE)
        share0, share1 = isp_ed_enc(secret_img=secret_img,
                                    alpha=args.alpha,
                                    gamma=args.gamma,
                                    threshold=args.threshold)
        if args.share0:
            cv2.imwrite(args.share0, share0)
        if args.share1:
            cv2.imwrite(args.share1, share1)
    else:
        share0 = cv2.imread(args.share0, cv2.IMREAD_GRAYSCALE)
        share1 = cv2.imread(args.share1, cv2.IMREAD_GRAYSCALE)

    if args.dec:
        dec = isp_ed_dec(share0, share1)
        cv2.imwrite(args.dec, dec)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--secret')
    parser.add_argument('--share0')
    parser.add_argument('--share1')
    parser.add_argument('--dec')
    parser.add_argument('--dec-only', action='store_true')

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    main(parser.parse_args())
