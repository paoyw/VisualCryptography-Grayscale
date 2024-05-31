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

rng = np.random.default_rng()

# def adjsut_block( block ):
#     '''
#     Create the share block according the paramater setting.
#     '''
    
#     _cor = rng.choice( np.prod(block_size), size = black_pixel_num, replace = False )
#     cor_row, cor_col = np.unravel_index( _cor, block_size )

#     new_block = np.zeros( block_size, dtype = 'uint8' )

#     cor_list = list( zip(cor_row, cor_col) )

#     for cor in cor_list:
#         new_block[cor] = 1

#     return new_block

def adjust_block(block, p_w, p_b):
    """
    Adjusts the number of black pixels in a block to either p_w or p_b.
    
    Args:
        block (numpy.ndarray): Binary block
        p_w (int): Target number of black pixels for white blocks
        p_b (int): Target number of black pixels for black blocks
        
    Returns:
        numpy.ndarray: Adjusted binary block
    """
    p = np.sum(block)
    # m = block.shape[0]
    
    if p > np.prod(block.shape) // 2:
        # Black block
        block[:] = 0
        _block = block.ravel()
        _block[:p_b] = 1
        rng.shuffle(_block)
        block = _block.reshape( block.shape )
    else:
        # White block
        block[:] = 0
        _block = block.ravel()
        _block[:p_w] = 1
        rng.shuffle(_block)
        block = _block.reshape( block.shape )
        
    return block

def generate_shares(secret_block, share_blocks, p_siw, p_sib, p_shw, p_shb, d, n):
    """
    Generates meaningful shares for (2, n) or (n, n) visual cryptography scheme.
    
    Args:
        secret_block (numpy.ndarray): Binary block from secret image
        share_blocks (list): List of binary blocks from share images
        p_siw (int): Number of black pixels in white block of secret image
        p_sib (int): Number of black pixels in black block of secret image
        p_shw (int): Number of black pixels in white block of share images
        p_shb (int): Number of black pixels in black block of share images
        d (int): Contrast parameter for staggering black pixels
        n (int): Number of shares
        
    Returns:
        list: List of generated share blocks as NumPy arrays
    """
    m = secret_block.shape[0]  # Block size
    secret_block = adjust_block(secret_block, p_siw, p_sib)
    p = np.sum(secret_block)  # Number of black pixels in secret block
    
    # Initialize share blocks
    share_blocks = [adjust_block(block, p_shw, p_shb) for block in share_blocks]
    res_share_blocks = []
    
    for i in range(1, n):
        p_prev = sum(np.sum(share) for share in share_blocks[:i])  # Number of black pixels in previous stack
        curr_share = share_blocks[i]
        p_curr = np.sum(share_blocks[i], dtype = 'uint8')  # Number of black pixels in current share
        curr_share = np.zeros_like(share_blocks[i])

        # Record locations of black, white, and full black pixels in previous stack
        prev_stack = np.sum(share_blocks[: i], axis = 0)
        b_prev = np.nonzero(prev_stack > 0)  # Locations of black pixels
        w_prev = np.nonzero(prev_stack == 0)  # Locations of white pixels
        # t_prev = np.nonzero(np.sum(share_blocks[: i], axis = 0) == i * p_shb)  # Locations of full black pixels
        t_prev = np.nonzero(np.sum(share_blocks[: i], axis = 0) == i ) 

        if len( b_prev[0] ) >= p:
            # Overlap all black pixels of current share on locations in b_prev
            curr_share[b_prev[0][:p_curr], b_prev[1][:p_curr]] = 1
            curr_share[w_prev] = 0
        else:
            # Stagger d black pixels of current share on locations in w_prev
            # curr_share[np.random.permutation(w_prev[0])[:d]] = 1
            curr_share[w_prev[0][:d], w_prev[1][:d]] = 1
            
            # Remaining (p_curr - d) black pixels on locations in t_prev
            curr_share[(t_prev[0][:p_curr - d], t_prev[1][:p_curr - d])] = 1
            
        share_blocks[i] = curr_share
        res_share_blocks.append(curr_share)

    return share_blocks

def generate_shares_2_2(secret_block, share1_block, share2_block, p_siw, p_sib, p_shw, p_shb):
    """
    Generates meaningful shares for (2, 2) visual cryptography scheme.
    
    Args:
        secret_block (numpy.ndarray): Binary block from secret image
        share1_block (numpy.ndarray): Binary block from first share image
        share2_block (numpy.ndarray): Binary block from second share image
        p_siw (int): Number of black pixels in white block of secret image
        p_sib (int): Number of black pixels in black block of secret image
        p_shw (int): Number of black pixels in white block of share images
        p_shb (int): Number of black pixels in black block of share images
        
    Returns:
        tuple: Tuple of generated share blocks as NumPy arrays
    """
    # Adjust input blocks
    secret_block = adjust_block(secret_block, p_siw, p_sib)
    share1_block = adjust_block(share1_block, p_shw, p_shb)
    share2_block = adjust_block(share2_block, p_shw, p_shb)
    
    p = np.sum(secret_block)  # Number of black pixels in secret block
    p1 = np.sum(share1_block)
    p2 = np.sum(share2_block)
    
    # Record locations of black and white pixels in share1
    b1 = np.nonzero(share1_block == 1)
    w1 = np.nonzero(share1_block == 0)
    
    share2_block = np.zeros_like(share2_block)

    # Generate share2 block
    # if p1 >= p:
    #     # Overlap p2 black pixels of share2 on locations in b1
    #     # share2_block[b1[0][:p2], b1[1][:p2]] = 1
    #     # share2_block[w1] = 0
    # else:
    # Stagger (p - p1) black pixels of share2 on locations in w1
    share2_block[w1[0][:p - p1], w1[1][:p - p1]] = 1
    
    # Remaining (p2 - (p - p1)) black pixels on locations in b1
    share2_block[b1[0][:(p2 - (p - p1))], b1[1][:(p2 - (p - p1))]] = 1
    
    return share1_block, share2_block

def split_into_blocks(img, block_size):
    # Get the dimensions of the image
    height, width = img.shape
    
    # Ensure the image dimensions are divisible by the block size
    if height % block_size[0] != 0 or width % block_size[1] != 0:
        raise ValueError("The image dimensions must be divisible by the block size.")
    
    # Calculate the number of blocks along each dimension
    num_blocks_h = height // block_size[0]
    num_blocks_w = width // block_size[1]
    
    # Split the image into blocks
    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = img[i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1]]
            blocks.append(block)
    
    return blocks

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    src_img_1 = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
    src_img_2 = cv2.imread('img2.png', cv2.IMREAD_GRAYSCALE)
    src_img_3 = cv2.imread('img3.png', cv2.IMREAD_GRAYSCALE)
    src_img_4 = cv2.imread('img4.png', cv2.IMREAD_GRAYSCALE)
    src_img_5 = cv2.imread('img5.png', cv2.IMREAD_GRAYSCALE)

    block_size = (2, 2)
    height, width = src_img_4.shape

    start = time.time()

    _src_img_1 = np.where( src_img_1 < 127, 1, 0 )
    _src_img_2 = np.where( src_img_2 < 127, 1, 0 )
    _src_img_3 = np.where( src_img_3 < 127, 1, 0 )
    _src_img_4 = np.where( src_img_4 < 127, 1, 0 )
    _src_img_5 = np.where( src_img_5 < 127, 1, 0 )

    i1l = split_into_blocks( _src_img_1, block_size )
    i2l = split_into_blocks( _src_img_2, block_size )
    i3l = split_into_blocks( _src_img_3, block_size )
    i4l = split_into_blocks( _src_img_4, block_size )
    i5l = split_into_blocks( _src_img_5, block_size )

    _l = list(zip( i1l, i2l, i3l, i4l, i5l ))

    p_siw, p_sib, p_shw, p_shb = 3, 4, 2, 3
    d, n = 1, 3

    res_img_list = []

    for _ in range(n):
        res_img_list.append(np.zeros_like( src_img_4 ))

    for index, blocks in enumerate(_l):
        res_blocks = generate_shares(
            secret_block = blocks[4] if args.secret_img == 'img5' else blocks[3],
            share_blocks = blocks[: 3],
            p_siw = p_siw,
            p_sib = p_sib, 
            p_shw = p_shw,
            p_shb = p_shb,
            d = d,
            n = n
        )

        # res_blocks = generate_shares_2_2(
        #     secret_block = blocks[4],
        #     share1_block = blocks[1],
        #     share2_block = blocks[2],
        #     p_siw = p_siw,
        #     p_sib = p_sib, 
        #     p_shw = p_shw,
        #     p_shb = p_shb
        # )

        # break
        i, j = np.unravel_index( index, (height // block_size[0], width // block_size[1]) )

        res_blocks = res_blocks[:n]
        
        for r_i, res in enumerate(res_blocks):
            res_img_list[r_i][ i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1] ] = res

    for _ in range(n):
        cv2.imwrite(f'res_{_ + 1}.png', 255 - res_img_list[_] * 255)

    end = time.time()
    print(f'Used_time: {end - start}')

    result_img = np.zeros_like( src_img_4 )
    for res in res_img_list:
        result_img |= res
    
    cv2.imwrite(f'stacked_res.png', 255 - result_img * 255)
