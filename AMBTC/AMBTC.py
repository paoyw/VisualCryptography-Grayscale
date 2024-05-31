# DIP Final Project
# Name: ZhiBao Lu
# ID #: R12922196
# email: r12922196@ntu.edu.tw
import cv2
import math
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import colorsys

from argparse import ArgumentParser

MAX_VALUE=2**8-1

BLOCK_SIZE = 4

def parse_arguments():
    parser = ArgumentParser(description="Digital Image Processing Final Project")
    
    parser.add_argument("--carry", default="SampleImages/", type=str, help="The path of input images which you wanna process.")
    parser.add_argument("--secret", default="SampleImages/", type=str, help="The path of secret images which you wanna process.")

    parser.add_argument("--shares", default="SampleImages/", type=str, help="The path of input images which you wanna process.")
    parser.add_argument("--output", default="output/", type=str,help="The path of output images which you wanna save.")
    parser.add_argument("--debug",  action="store_true", help="Debug Mode")

    parser.add_argument("--encrypt", action="store_true", help="Encrypt the image.")    
    parser.add_argument("--reconstruct", action="store_true", help="Reconstruct the  secret image.")    
    parser.add_argument("--decrypt", action="store_true", help="Decrypt the image.")
    args = parser.parse_args()
    
    return args


def randBP():
    return np.random.randint(2, size=(BLOCK_SIZE, BLOCK_SIZE))

def ambtcoding(img):
    m, n = img.shape

    bitmap = np.zeros((m, n), dtype=np.uint8)

    a_i = np.zeros((m//BLOCK_SIZE, n//BLOCK_SIZE), dtype=np.uint8)
    b_i = np.zeros((m//BLOCK_SIZE, n//BLOCK_SIZE), dtype=np.uint8)
    all_one = BLOCK_SIZE*BLOCK_SIZE
    for i in range(0, m, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            xbar = np.mean(block)
            outB = (block >= xbar)

            k  =np.sum(outB)
            b_i[i//BLOCK_SIZE ,j//BLOCK_SIZE] = np.average(block[outB])
            if k == all_one:
                a_i[i//BLOCK_SIZE ,j//BLOCK_SIZE] = np.average(block[outB])
            else:
                a_i[i//BLOCK_SIZE ,j//BLOCK_SIZE] = np.average(block[~outB])
            bitmap[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = outB
        
    return (a_i, b_i, bitmap)

# Avoiding the false judgement during the reconstructing procedure
def preprocess(a_i, b_i, bitmap):
    m, n = a_i.shape
    for i in range(m):
        for j in range(n):
            if a_i[i, j] == b_i[i, j]:
                rb = randBP()
                bitmap[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = rb 

    return bitmap

# encrypt : Construct two shares for the proposed (2, 2) scheme with reversibility

def encrypt(secret, ambtc_1, ambtc_2):
    """
    Construct two shares for the proposed (2, 2) scheme with reversibility.

    Parameters:
    Secret (numpy.ndarray): Secret image of size (M, M).
    ambtc_1 (numpy.ndarray): Modified AMBTC-compressed codes. eg(ai , bi, bitmap).
    ambtc_2 (numpy.ndarray): Modified AMBTC-compressed codes. eg(ai , bi, bitmap).

    Returns:
    AS (numpy.ndarray): First meaningful shadow.
    BS (numpy.ndarray): Second meaningful shadow.
    """
    m, n = secret.shape
    assert (m, n) == ambtc_1[0].shape, "The size of secret image should be same with shadows images."

    AS = (ambtc_1[0].copy(), ambtc_1[1].copy(), ambtc_1[2].copy())
    BS = (ambtc_2[0].copy(), ambtc_2[1].copy(), ambtc_2[2].copy())

    for i in range(m):
        for j in range(n):
            # case 1: If the shared pixel Secret_{ij} is black
            if secret[i, j] == 0:
                # AS flipping 
                if np.random.rand() < 0.5:
                    AS[0][i, j], AS[1][i, j] = AS[1][i, j], AS[0][i, j]
                    AS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = \
                        1- AS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
                # BS flipping
                else:                    
                    BS[0][i, j], BS[1][i, j] = BS[1][i, j], BS[0][i, j]
                    BS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = \
                        1- BS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]

            # case 2: If the shared pixel SIi;j is white, the two related trios are
            # together to be flipped or not with 50% probability. 
            else:
                # Flipping together
                if np.random.rand() < 0.5:
                    # AS flippping
                    AS[0][i, j], AS[1][i, j] = AS[1][i, j], AS[0][i, j]
                    AS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = \
                        1- AS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
                    # BS flipping
                    BS[0][i, j], BS[1][i, j] = BS[1][i, j], BS[0][i, j]
                    BS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = \
                        1- BS[2][i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
    
    return AS, BS


def decrypt_ambtcoding(a_i, b_i , hb):
    m, n = a_i.shape
    result = np.zeros(hb.shape, dtype=np.uint8)
    all_one = BLOCK_SIZE*BLOCK_SIZE

    for i in range(m):
        for j in range(n):
            block = hb[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
            recover_block =  a_i[i, j]*(1-block) + b_i[i, j]* block
            result[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = recover_block
            
    return result    

# Reconstructing 1: stacking operation
def reconstruction_stacking(share1, share2):
    m, n =share1[2].shape
    shares = np.zeros((m, n, 2), dtype=np.uint8)
    shares[:,:, 0], shares[:,:, 1] = share1[2], share2[2]

    result =stacking(shares)

    return result*MAX_VALUE

# Reconstructing 2: Algorithm 3. Completely reconstruct the secret image
def reconstruction_xor(share1, share2):

    bitmap1, bitmap2 = share1[2], share2[2]
    HR = cv2.bitwise_xor(bitmap1, bitmap2)
    m, n = HR.shape
    result = np.zeros((m//BLOCK_SIZE, n//BLOCK_SIZE))

    for i in range(0, m, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            Hblock = HR[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            result[i//BLOCK_SIZE, j//BLOCK_SIZE] = 0 if np.sum(Hblock) == BLOCK_SIZE*BLOCK_SIZE  else 1
    

    return (result*MAX_VALUE).astype(np.uint8)

# Restoring procedure is to retrieve an original cover compressed codes without distortion
def decrypt_share(ambtcoding_image):
    m, n = ambtcoding_image[2].shape
    assert m == n, "Share image should be squared image.(eg. size M by M )"

    bitmap = ambtcoding_image[2].copy()
    a_i = ambtcoding_image[0].copy()
    b_i = ambtcoding_image[1].copy()
    for i in range(0, m, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            rescale_i, rescale_j = i//BLOCK_SIZE, j//BLOCK_SIZE
            # If ai;j > bi;j, the not Logical operation is applied on HBi;j, that
            # swap
            if a_i[rescale_i, rescale_j] > b_i[rescale_i, rescale_j]:
                a_i[rescale_i, rescale_j], b_i[rescale_i, rescale_j] = \
                    b_i[rescale_i, rescale_j], a_i[rescale_i, rescale_j]
                bitmap[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 1- bitmap[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

    return a_i, b_i, bitmap

# encode shares images
def encode_share(share):
    '''
    Parameters:
    Share (numpy.ndarray): Share image

    Returns:
    AS (numpy.ndarray): AMBTC compressed image.
    '''

    m, n = share.shape
    assert m == n, "Share image should be squared image.(eg. size M by M )"

    bitmap = np.zeros((m, n), dtype=np.uint8)
    a_i = np.zeros((m//BLOCK_SIZE, n//BLOCK_SIZE), dtype=np.uint8)
    b_i = np.zeros((m//BLOCK_SIZE, n//BLOCK_SIZE), dtype=np.uint8)

    for i in range(0, m, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            block = share[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            high = np.max(block)
            low = np.min(block)
            a_i[i//BLOCK_SIZE, j//BLOCK_SIZE] = low
            b_i[i//BLOCK_SIZE, j//BLOCK_SIZE] = high 
            bitmap[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE][block==high] = 1

    
    return (a_i, b_i, bitmap)


# stacking operation
def stacking(shares):

    m, n, numOfShares = shares.shape

    secret = shares[:, : , 0].copy() 

    for idx in range(1, numOfShares):
        secret = secret & shares[:,:, idx]
    return secret    

def pack_bitmap(image):
    m, n = image.shape
    result = np.zeros((m, int(np.ceil(n/8))) )
    for j in range(0, n, 8):
        split = image[:, j:j+8]
        split = np.packbits(split)
        result[:, j//8] = split
    return result

def unpack_bitmap(image):
    m, n = image.shape
    n = n*8
    result = np.zeros((m, n) )
    for j in range(0, n, 8):
        split = image[:, j//8].reshape(m, -1)
        split = np.unpackbits(split, axis=1) 
        result[:, j:j+8] = split
    return result

if __name__ == "__main__":

    args = parse_arguments()
    # [Encrypt]
    if args.encrypt:
        gray = cv2.imread(args.carry, cv2.IMREAD_GRAYSCALE)
        secret = cv2.imread(args.secret, cv2.IMREAD_GRAYSCALE)
    
        # Preprocesing for avoiding the false judgement during the reconstructing procedure
        a_i, b_i, hb = ambtcoding(gray)
        hb = preprocess(a_i, b_i, hb)
        
        # Testing: build ambtc image
        # recover_img = decrypt_ambtcoding(a_i, b_i, hb)
        # cv2.imwrite(os.path.join(args.output, "ambtc_recover.png"), recover_img)
        ambtc_1 = (a_i , b_i, hb)
        ambtc_2 = (a_i.copy(), b_i.copy(), hb.copy())
        share1, share2 = encrypt(secret, ambtc_1, ambtc_2)
        
        # Save Share
        shadow1 = decrypt_ambtcoding(*share1)
        cv2.imwrite(os.path.join(args.output, "share1.png"), shadow1)
        shadow2 = decrypt_ambtcoding(*share2)
        cv2.imwrite(os.path.join(args.output, "share2.png"), shadow2)
        

        # Save AMBTC image 
        #cv2.imwrite(os.path.join(args.output, "share1.bmp"), share1[2])
        #cv2.imwrite(os.path.join(args.output, "share2.bmp"), share2[2])
        
        share1_bitmap = pack_bitmap(share1[2])
        share2_bitmap = pack_bitmap(share2[2])
        cv2.imwrite(os.path.join(args.output, "share1_bitmap.png"), share1_bitmap)
        cv2.imwrite(os.path.join(args.output, "share2_bitmap.png"), share2_bitmap)
        cv2.imwrite(os.path.join(args.output, "share1_ai.png"), share1[0])
        cv2.imwrite(os.path.join(args.output, "share1_bi.png"), share1[1])
        cv2.imwrite(os.path.join(args.output, "share2_ai.png"), share2[0])
        cv2.imwrite(os.path.join(args.output, "share2_bi.png"), share2[1])


    if args.reconstruct:
        # Reconstructing preocedure: two ways to
        # reconstruct the secret image by qualified participants
        # Read AMBTC image
        shares = []
        for i in range(2):
            a_i = cv2.imread(os.path.join(args.shares, "share{}_ai.png".format(i+1)), cv2.IMREAD_GRAYSCALE)
            b_i = cv2.imread(os.path.join(args.shares, "share{}_bi.png".format(i+1)), cv2.IMREAD_GRAYSCALE)
            bitmap = cv2.imread(os.path.join(args.shares, "share{}_bitmap.png".format(i+1)), cv2.IMREAD_GRAYSCALE)

            bitmap = unpack_bitmap(bitmap)
            shares.append((a_i, b_i, bitmap))
        share1, share2 = shares

        # First method: stacking operation
        reconstruct_stacking = reconstruction_stacking(share1, share2)
        cv2.imwrite(os.path.join(args.output, "reconstruct_stacking.png"), reconstruct_stacking)
        
        # Second method: xor operation
        reconstruct_xor = reconstruction_xor(share1, share2)
        cv2.imwrite(os.path.join(args.output, "reconstruct_xor.png"), reconstruct_xor)



    if args.decrypt:
        
        # Read AMBTC image
        shares = []
        for i in range(2):
            a_i = cv2.imread(os.path.join(args.shares, "share{}_ai.png".format(i+1)), cv2.IMREAD_GRAYSCALE)
            b_i = cv2.imread(os.path.join(args.shares, "share{}_bi.png".format(i+1)), cv2.IMREAD_GRAYSCALE)
            bitmap = cv2.imread(os.path.join(args.shares, "share{}_bitmap.png".format(i+1)), cv2.IMREAD_GRAYSCALE)

            bitmap = unpack_bitmap(bitmap)
            shares.append((a_i, b_i, bitmap))
        share1, share2 = shares
        # Decrypt ambtc-compreesed image
        decrypt_share1 = decrypt_share(share1)
        decrypt_share2 = decrypt_share(share2)

        cv2.imwrite(os.path.join(args.output, "decrypt_bitmap1.png"), decrypt_share1[2]*MAX_VALUE)
        cv2.imwrite(os.path.join(args.output, "decrypt_bitmap2.png"), decrypt_share2[2]*MAX_VALUE)


        decrypt_share1 = decrypt_ambtcoding(*decrypt_share1)
        decrypt_share2 = decrypt_ambtcoding(*decrypt_share2)
            
        cv2.imwrite(os.path.join(args.output, "decrypt_carry1.png"), decrypt_share1)
        cv2.imwrite(os.path.join(args.output, "decrypt_carry2.png"), decrypt_share2)


