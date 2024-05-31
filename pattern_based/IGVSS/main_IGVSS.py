import numpy as np
import cv2
from pathlib import Path
import copy
import time
from scipy.ndimage import median_filter

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = 'my description')
    parser.add_argument('--share_img_path', type = str)
    parser.add_argument('--secret_img_path', type = str)
    return parser

def error_diffusion( src_img, mode = 'floyd' ):
    if mode == 'floyd':
        kernel = np.array([[0, 0, 7], [3, 5, 1]], dtype = 'float64') / 16
        pad = 1
    elif mode == 'jarvis':
        kernel = np.array( [[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]] ) / 48
        pad = 2
    elif mode == 'stucki':
        kernel = np.array( [[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]] ) / 42
        pad = 2
    elif mode == 'atkinson':
        kernel = np.array( [[0, 0, 0, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]] ) / 8
        pad = 2
    elif mode == 'sierra':
        kernel = np.array( [[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]] ) / 32
        pad = 2
    
    height, width = src_img.shape
    
    src_img = np.lib.pad(src_img, (pad, pad), 'constant')

    # Preprocess.
    src_img = src_img.astype('float64')
    src_img /= 255
    threshold = 0.5

    # Apply serpentine scanning.
    for i in range(pad, height + pad):
        if not (i % 2):
            for j in range(pad, width + pad):
                f = src_img[i, j]
                g = np.where( f > threshold, 1, 0 )
                e = f - g
                temp = e * kernel
                src_img[i : i + pad + 1, j - pad: j + pad + 1] += temp
        else:
            for j in range(width + pad - 1, pad - 1, -1):
                f = src_img[i, j]
                g = np.where( f > threshold, 1, 0 )
                e = f - g
                temp = e * np.flip(kernel, axis = 1)
                src_img[i : i + pad + 1, j - pad: j + pad + 1] += temp
    
    new_img = np.where(src_img > threshold, 255, 0)[pad : height + pad, pad : width + pad]
    return new_img

def shares_construction( HI, MVI1, MVI2 ):
    n, m = HI.shape
    # share1, share2 = np.zeros( (n, 2 * m), dtype = 'int16' ), np.zeros( (n, 2 * m), dtype = 'int16' )
    share1, share2 = np.zeros( (n, m + 1), dtype = 'int16' ), np.zeros( (n, m + 1), dtype = 'int16' )
    k = 1
    # It seems like the original algorithm is wrong.
    # for i in range( n ):
    #     for j in range( m ):
    #         if HI[i, j] == 255:
    #             share1[i, j] = MVI1[i, j]
    #             share1[i, j + 1] = MVI1[i, j] + k
    #             share2[i, j] = MVI2[i, j]
    #             share2[i, j + 1] = MVI1[i, j] + k
    #         elif HI[i, j] == 0:
    #             share1[i, j] = MVI1[i, j]
    #             share1[i, j + 1] = MVI1[i, j] + k
    #             share2[i, j] = MVI2[i, j] + k
    #             share2[i, j + 1] = MVI1[i, j]
    # for i in range( n ):
    #     for j in range( m ):
    #         if HI[i, j] == 255:
    #             share1[i, 2 * j] = MVI1[i, j]
    #             share1[i, 2 * j + 1] = MVI1[i, j] + k
    #             share2[i, 2 * j] = MVI2[i, j]
    #             share2[i, 2 * j + 1] = MVI2[i, j] + k
    #         elif HI[i, j] == 0:
    #             share1[i, 2 * j] = MVI1[i, j]
    #             share1[i, 2 * j + 1] = MVI1[i, j] + k
    #             share2[i, 2 * j] = MVI2[i, j] + k
    #             share2[i, 2 * j + 1] = MVI2[i, j]
    for i in range( n ):
        for j in range( m ):
            if HI[i, j] == 255:
                share1[i, j] = MVI1[i, j]
                share1[i, j + 1] = MVI1[i, j] + k
                share2[i, j] = MVI2[i, j]
                share2[i, j + 1] = MVI1[i, j] + k
            elif HI[i, j] == 0:
                share1[i, j] = MVI1[i, j]
                share1[i, j + 1] = MVI1[i, j] + k
                share2[i, j] = MVI2[i, j] + k
                share2[i, j + 1] = MVI1[i, j]
    return share1, share2

def encode_binary_pattern(binary_pattern):
    """
    Encodes a 4x4 binary pattern into an index using Equation (1)
    """
    index = 0
    for pixel in np.ravel(binary_pattern):
        index = (index << 1) | pixel
    return index

def find_edge_type(edge_pattern, elut):
    """
    Finds the edge type index for a given 4x4 edge pattern
    by searching in the ELUT edge patterns
    """
    for edge_type, pattern in enumerate(elut.edge_patterns):
        if np.array_equal(edge_pattern, pattern):
            return edge_type
    return -1  # Edge type not found

def best_linear_estimator(binary_pattern, neighboring_values):
    """
    Implements the best linear estimator for missing ELUT entries
    """
    # Let's number all the patterns which exist in the sample halftone images
    existing_patterns = []
    existing_contone_values = []
    for i, pat in enumerate(elut.edge_patterns):
        if np.any(elut.table[:,i] != 0):
            existing_patterns.append(pat)
            existing_contone_values.append(elut.table[np.nonzero(elut.table[:,i]), i])
    
    num_existing = len(existing_patterns)
    
    # Define the pattern matrix P and LUT vector y
    P = np.zeros((num_existing, binary_pattern.size), dtype=int)
    y = np.array(existing_contone_values)
    
    for i, pat in enumerate(existing_patterns):
        P[i] = pat.ravel()
    
    # Solve P @ x = y using least squares
    x = np.linalg.lstsq(P, y, rcond=None)[0]
    
    # Estimate the contone value for the given binary_pattern
    estimated_value = np.dot(binary_pattern.ravel(), x)
    
    # Clip and round the estimated value
    if estimated_value < 0:
        estimated_value = 0
    elif estimated_value > 255:
        estimated_value = 255
    else:
        estimated_value = round(estimated_value)
    
    return int(estimated_value)

# def best_linear_estimator(binary_pattern, neighboring_values):
#     """
#     Implements the best linear estimator for missing ELUT entries
#     """
#     # Your implementation of the best linear estimator goes here
#     # This is just a placeholder, replace with your actual code
#     return 128  # Returning a default value of 128 for simplicity

class ELUT:
    def __init__(self):
        self.table = np.zeros((65536, 39), dtype=np.uint8)
        self.edge_patterns = []
        self._initialize_edge_patterns()

    def _initialize_edge_patterns(self):
        # Category 1: Regular edge types
        self.edge_patterns.extend([np.zeros((4, 4), dtype=np.uint8)] * 34)
        
        # Horizontal edge patterns (indices 0-11)
        for i in range(12):
            pattern = np.zeros((4, 4), dtype=np.uint8)
            pattern[1:3, :] = 1
            self.edge_patterns[i] = np.rot90(pattern, i // 3)
        
        # Vertical edge patterns (indices 12-23)
        for i in range(12, 24):
            pattern = np.zeros((4, 4), dtype=np.uint8)
            pattern[:, 1:3] = 1
            self.edge_patterns[i] = np.rot90(pattern, (i - 12) // 3)
        
        # Diagonal edge patterns (indices 24-29)
        pattern = np.eye(4, dtype=np.uint8)
        self.edge_patterns[24] = pattern
        self.edge_patterns[25] = np.fliplr(pattern)
        self.edge_patterns[26] = np.rot90(pattern, 2)
        self.edge_patterns[27] = np.rot90(np.fliplr(pattern), 2)
        self.edge_patterns[28] = np.fliplr(self.edge_patterns[26])
        self.edge_patterns[29] = np.fliplr(self.edge_patterns[27])
        
        # Corner edge patterns (indices 30-33)
        pattern = np.zeros((4, 4), dtype=np.uint8)
        pattern[0, 0] = pattern[3, 3] = 1
        self.edge_patterns[30] = pattern
        self.edge_patterns[31] = np.rot90(pattern, 1)
        self.edge_patterns[32] = np.rot90(pattern, 2)
        self.edge_patterns[33] = np.rot90(pattern, 3)
        
        # Category 2: Irregular edge type
        self.edge_patterns.extend([np.ones((4, 4), dtype=np.uint8)])  # Index 34
        self.edge_patterns.extend([np.random.randint(0, 2, (4, 4), dtype=np.uint8) for _ in range(4)])  # Indices 35-38

    def build_up(self, training_images, training_halftones, lih):
        for gi, hi in zip(training_images, training_halftones):
            base_gray = lih(hi)
            edge_map = generate_edge_map(base_gray)
            
            for i in range(base_gray.shape[0] - 3):
                for j in range(base_gray.shape[1] - 3):
                    binary_window = hi[i:i+4, j:j+4].astype(int)
                    edge_pattern = edge_map[i:i+4, j:j+4]
                    
                    index = encode_binary_pattern(binary_window)
                    edge_type = find_edge_type(edge_pattern, self)
                    
                    if self.table[index, edge_type] == 0:
                        self.table[index, edge_type] = gi[i+1:i+3, j+1:j+3].mean()

def generate_edge_map(base_gray_image):
    """
    Generates the edge map by applying the Canny edge detector
    to the base gray image
    """
    # Apply Canny edge detector
    edge_map = cv2.Canny(base_gray_image, 100, 200)
    return edge_map

def revealing_phase(shares, base_gray_image, elut):

    """
    Performs the revealing phase of the IGVSS scheme
    using the edge-based LUT (ELUT) approach
    """
    # Step 1: Generate the edge map from the base gray image
    edge_map = generate_edge_map(base_gray_image)

    # Step 2: Get halftone image HI' by XORing share images S1 and S2
    hi_prime = shares[0] ^ shares[1]

    # Step 3: Initialize reconstructed grayscale image GI'
    gi_prime = np.zeros_like(hi_prime, dtype=np.uint8)

    # Step 4: Slide 4x4 window over HI' and reconstruct GI' using ELUT
    window_size = 4
    for i in range(0, hi_prime.shape[0] - window_size + 1, window_size):
        for j in range(0, hi_prime.shape[1] - window_size + 1, window_size):
            print(f'i, j: {i, j}')
            # Get 4x4 binary subimage b from W
            binary_window = hi_prime[i:i+window_size, j:j+window_size]
            binary_pattern = binary_window.astype(int)

            # Encode b to get index i
            index = encode_binary_pattern(binary_pattern)

            # Get corresponding 4x4 edge pattern subimage e from edge map
            edge_pattern = edge_map[i:i+window_size, j:j+window_size]

            # Find edge type j by searching e in ELUT edge patterns
            edge_type = find_edge_type(edge_pattern, elut)

            # If (i,j) exists in ELUT, reconstruct using mean value
            if edge_type != -1 and elut.table[index, edge_type] != 0:
                mean_value = elut.table[index, edge_type]
                gi_prime[i:i+window_size, j:j+window_size] = mean_value

            # Else handle missing (i,j) entries
            else:
                # Phase 1: If all 0s, set 0; if all 1s, set 255
                if np.all(binary_pattern == 0):
                    gi_prime[i:i+window_size, j:j+window_size] = 0
                elif np.all(binary_pattern == 1):
                    gi_prime[i:i+window_size, j:j+window_size] = 255

                # Phase 2: Else use best linear estimator
                else:
                    estimated_value = best_linear_estimator(binary_pattern, gi_prime[max(0, i-window_size):i, j:j+window_size])
                    gi_prime[i:i+window_size, j:j+window_size] = estimated_value

    return gi_prime

def inverse_halftoning(halftone_img):
    # kernel = np.ones((3, 3), np.uint8)
    halftone_img = halftone_img.astype( np.uint8 )
    filtered_image = median_filter(halftone_img, size=3)
    return filtered_image

if __name__ == '__main__':
    # pass

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # secret = cv2.imread('misc/7.1.09.tiff', cv2.IMREAD_GRAYSCALE)
    # MVI1 = cv2.imread('misc/7.1.10.tiff', cv2.IMREAD_GRAYSCALE)
    # MVI2 = cv2.imread('misc/7.1.10.tiff', cv2.IMREAD_GRAYSCALE)

    secret = cv2.imread(args.secret_img_path, cv2.IMREAD_GRAYSCALE)
    MVI1 = cv2.imread(args.share_img_path, cv2.IMREAD_GRAYSCALE)
    MVI2 = cv2.imread(args.share_img_path, cv2.IMREAD_GRAYSCALE)

    print(secret.shape)
    print(secret.max(), secret.min())
    print(MVI1.shape)
    print(MVI1.max(), MVI1.min())
    print(MVI2.shape)
    print(MVI2.max(), MVI2.min())

    _secret = copy.deepcopy(secret)

    start = time.time()

    # Now do the task.
    # Do the error diffusion for secret image.
    HI = error_diffusion( _secret )
    cv2.imwrite(f'HI.png', HI)

    share1, share2 = shares_construction( HI, MVI1, MVI2 )
    cv2.imwrite(f'share_1.png', share1)
    print(share1)
    cv2.imwrite(f'share_2.png', share2)
    print(share2)
    HI_prime = np.abs(share1 - share2)
    print(HI_prime)
    print(HI_prime.min(), HI_prime.max())

    HI_prime = 255 - 255 * HI_prime
    HI_prime = HI_prime[:secret.shape[0], :secret.shape[1]]

    cv2.imwrite(f'HI_prime.png', HI_prime)

    end = time.time()
    print(f'Used_time: {end - start}')

    _secret = copy.deepcopy(secret)
    cv2.imwrite(f'_secret.png', _secret)
    
    output = cv2.GaussianBlur(HI_prime, (3, 3), 0)
    # output = cv2.blur(HI_prime, (2, 2))
    # print(output)
    # print(output.min(), output.max())
    cv2.imwrite(f'output.png', output)
