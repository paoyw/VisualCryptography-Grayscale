import numpy as np


def errorDiffusionFloydSteinberg(img: np.array, threshold: int = 128):
    """
    Apply the Flayd-Steinberg method on the input images.

    Inputs:
        img: The input image.

    Returns:
        output: The halftone result.
    """
    h, w = img.shape
    error_img = np.zeros((h + 1, w + 2), dtype=np.float32)
    error_img[:h, 1:w + 1] = img.astype(np.float32)
    result = np.zeros((h, w), dtype=np.float32)
    weights = np.array([
        [0, 0, 7 / 16],
        [3 / 16, 5 / 16, 1 / 16]
    ])
    for i in range(h):
        for j in range(w):
            if error_img[i][j + 1] > threshold:
                result[i][j] = 255
            else:
                result[i][j] = 0
            error_img[i:i + 2, j:j + 3] += \
                weights * (error_img[i][j + 1] - result[i][j])
    return result.astype(np.uint8)
