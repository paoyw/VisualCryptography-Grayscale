# DIP Final Project
# Name: ZhiBao Lu
# ID #: R12922196
# email: r12922196@ntu.edu.tw
import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import colorsys
from scipy.cluster.vq import vq, kmeans, whiten
import re

from argparse import ArgumentParser

MAX_VALUE=2**8-1

def parse_arguments():
    parser = ArgumentParser(description="Digital Image Processing Final Project")
    
    parser.add_argument("--secret", default="SampleImages/", type=str, help="The path of input images which you wanna process.")
    parser.add_argument("--shares", default="SampleImages/", type=str, help="The path of input images which you wanna process.")
    parser.add_argument("--output", default="output/", type=str,help="The path of output images which you wanna save.")
    parser.add_argument("--debug",  action="store_true", help="Debug Mode")

    parser.add_argument('--m', type=int, default=3 , help='The total number of shares')
    parser.add_argument('--k', type=int, default=3 ,help='The threshold number of shares to reconstruct the image')
    parser.add_argument('--num_decrypt', type=int, default=3 ,help="The number of shares be decrypted")
    

    parser.add_argument("--encrypt", action="store_true", help="Encrypt the image.")    
    parser.add_argument("--decrypt", action="store_true", help="Decrypt the image.")
    parser.add_argument("--decompress", action="store_true", help="Fecompress the image.")
    args = parser.parse_args()
    
    return args

# (2, 2)-QSP based VCS of grayscale image
def QSP_encrypt(gray):
    assert len(gray.shape) == 2, "Please give me the orignal images."

    m, n = gray.shape

    z = np.random.uniform(low=0, high=1, size=(m, n))
    u = np.random.uniform(low=0, high=1, size=(m, n))

    normal_gray = (gray-np.min(gray))/(np.max(gray)-np.min(gray))

    share1, share2 = np.zeros((m, n)), np.zeros((m, n))
    for row in range(m):    
        for col in range(n):
            intensity = normal_gray[row, col]
            if intensity >= 0.5:
                # case 1
                if z[row, col] <= 0.5:
                    share1[row, col], share2[row,col] = u[row, col], u[row, col]
                else:
                    share1[row, col], share2[row,col] = 1-u[row, col], 1-u[row, col]
            else:
                # case 2
                if z[row, col] <= 0.5:
                    share1[row, col], share2[row,col] = u[row, col], 1-u[row, col]
                else:
                    share1[row, col], share2[row,col] = 1-u[row, col], u[row, col]

    return [share1, share2]




# Decrypt image
def stacking(shares):

    m, n, numOfShares = shares.shape

    secret = shares[:, : , 0].copy() 

    for idx in range(1, numOfShares):
        secret = secret & shares[:,:, idx]
    return secret


# Utility for Shamir scheme
def get_prime_numbers(max_number):
    primers = [2]
    for num in range(3, max_number):
        now_iter  = 2 

        while now_iter**2 <= num:
            if ((num%now_iter) == 0) :
                break 
            now_iter += 1
        if now_iter**2 > num:
            primers.append(num)
    return primers


# Utility for Shamir scheme
def lagrange(x:list, y:list, k:int):
    assert len(x) == len(y) , "[Lagrange Interpolation] The size of x and y should be same."
    l = np.zeros(len(x))
    for i in range(k):
        l[i] = 1
        for j in range(k):
            if i != j:
                numer =  -1*x[j]
                denom = (x[i] - x[j])
                l[i] = l[i]*numer/denom
    return np.sum(np.multiply(l, y))



# Vector Quantization: Hilbert Scan
def hilbert_order(ORDER):            
    N = 2**ORDER                                     
    P = np.zeros((2, N * N)).astype(np.uint8)                        
    for i in range(N * N):
        U = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])   
        V = np.array([0, 0])                  
        for j in reversed(range(ORDER)):              
            ID = i // 4**j % 4                        
            L = 2**j                                 
            if   ID == 0:
                U[1], U[3] = U[3].copy(), U[1].copy() 
            elif ID == 3:
                U[0], U[2] = U[2].copy(), U[0].copy() 
            V += U[ID] * L                          
        P[:, i] = V                                                                       
    return P                         

# Vector Quantization: Hilbert Scan
def hilbertScan(img, block_size=8):
    order = int(np.log2(block_size))
    pos_x, pos_y = hilbert_order(ORDER=order)
    
    m, n = img.shape
    result= np.zeros((m//block_size, n*block_size)).astype(np.uint8)
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            block = img[i:i+block_size, j:j+block_size]
            vector = np.zeros(len(pos_x)).astype(np.uint8)
            for iidx in range(len(pos_x)):
                vector[iidx] = block[pos_y[iidx], pos_x[iidx]]
            save_start, save_end = (j//block_size)*len(pos_x), (j//block_size+1)*len(pos_x)
            result[i//block_size, save_start:save_end] = vector
    return result



def decrypt_hilbert(img, block_size=8):
    m, n = img.shape
    m, n = m*block_size, n//block_size
    order = int(np.log2(block_size))
    pos_x, pos_y = hilbert_order(ORDER=order)
    inverse_hilbert_image = np.zeros((m, n), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(0, img.shape[1], block_size*block_size):
            vector = img[i, j:j+block_size*block_size]
            block = np.zeros((block_size, block_size), dtype=np.uint8)
            for idx in range(len(pos_x)):
                block[pos_y[idx], pos_x[idx]] = vector[idx]
            start_row = i*block_size
            start_col = j//block_size
            inverse_hilbert_image[start_row:start_row+block_size, start_col:start_col+block_size] = \
                    block

    return inverse_hilbert_image


def vq_compress(img, block_size=4, vector_bits=6):
    codebook_size = 2**vector_bits
    m, n = img.shape
    
    vector_dimens = block_size*block_size

    img_vectors = []

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            img_vectors.append(img[i:i+block_size, j:j+block_size].flatten())

    img_vectors = np.array(img_vectors).astype(np.float64)
    number_of_image_vectors = img_vectors.shape[0]
    # Create ecentorid
    
    centroids = 255 * np.random.rand(codebook_size, vector_dimens)
    reconstruction_values, distortion =kmeans(img_vectors, centroids, iter=30)

    image_vector_indices, distance = vq(img_vectors, reconstruction_values)
    codebook = dict()
    codebook["image"] = str(np.array([m, n]))
    codebook["vector"] = reconstruction_values
    codebook["indices"] = image_vector_indices.astype(np.uint8)

    return codebook


# Compression using Vector Quantization
def vector_quantization(shares):
    
    num_shares = len(shares)
    codebook = dict()
    
    for idx in range(num_shares):
        share = shares[idx]    
        hilbert_share = hilbertScan(share)
        cb = vq_compress(hilbert_share)
        if idx == 0:
            codebook['image'] = cb['image']
        codebook['vector{}'.format(idx)] = cb['vector']
        codebook['indices{}'.format(idx)] = cb['indices']
    
    return codebook

def saveCodebook(path, codebook, m):
    with open(path, 'w+') as f:
        f.write(codebook["image"])
        f.write("\n")
        for idx in range(m):
            f.write(str(codebook["vector{}".format(idx)].shape)+"\n")      
            np.savetxt(f, codebook["vector{}".format(idx)], fmt='%.3f')  
            np.savetxt(f, codebook["indices{}".format(idx)], fmt="%d", newline=',')   
            f.write("\n")         


def read_codebook(path, m):
    codebook = dict()
    intPattern = "\d+"
    floatPattern ="\d+.\d+"
    with open(path, 'r') as f:
        lines = f.readlines()
    h, n = re.findall(intPattern, lines[0])
    h, n = int(h), int(n)
    codebook["image"] = [h, n]
    line_idx = 1
    for idx in range(m):
        centroids_size = list(map(lambda x: int(x), re.findall(intPattern, lines[line_idx])))
        centriods = []
        for j in range(centroids_size[0]):
            line_idx+= 1
            centriod_vec = list(map(lambda x: float(x), re.findall(floatPattern, lines[line_idx])))
            centriods.append(centriod_vec)
        line_idx += 1
        image_indices = list(map(lambda x: int(x), re.findall(intPattern, lines[line_idx])))
        
        line_idx += 1
        # Save Image indices and centriods into codebook
        codebook["vector{}".format(idx)] = np.array(centriods)        
        codebook["indices{}".format(idx)] = np.array(image_indices)

    return codebook

def vq_decompress(image_size, centriods_vec, image_vector_indices, block_size=4):
    m, n = image_size 
    result = np.zeros(image_size, dtype=np.uint8)

    for index in range(len(image_vector_indices)):
        start_row = int(index*block_size/n)*block_size
        end_row = start_row + block_size
        start_column = (index*block_size) % n
        end_column = start_column + block_size
        block = np.reshape(centriods_vec[image_vector_indices[index]],
                   (block_size, block_size))
        result[start_row:end_row, start_column:end_column] = block
    
    result = decrypt_hilbert(result)
    return result 


def decompress_codebook(codebook, m):
    height, width = codebook["image"]
    shares = []
    for idx in range(m):
        centroids_vec = codebook["vector{}".format(idx)] 
        image_indices = codebook["indices{}".format(idx)]
        share = vq_decompress((height,width), centroids_vec, image_indices)
        shares.append(share)

    return shares

def decompress_codebook_v2(codebook, m):
    height, width = codebook["image"]
    shares = []
    image_indices = codebook["indices"]
    stds = codebook["indices"]
    centroids_vec = codebook["vector"] 
    for i in range(height):
        for j in range(width):
            value =  image_indices[i*width+height] *stds
            print(value)

    return shares


# Shamir’s (k, m)-threshold scheme on each share (Encrypt)
def Shamir_encrypt(secret, k, m):
    h, n  = gray.shape
    
    coeff = np.random.randint(low=0, high=MAX_VALUE+1, size=(h, n, k-1))
    shares = []
    primer = np.array(get_prime_numbers(622))
    q =  primer[primer> secret.max()][0]
    for idx in range(1, m+1):
        factors = np.array([idx ** j for j in range(1, k)])
       
        fun_value = (np.matmul(coeff, factors) + secret) % q
        shares.append(fun_value)
    return shares

# Shamir’s (k, m)-threshold scheme on each share  (Decrypt)     
def Shamir_decrypt(shares, x, max_decrypt):
    m, n, numOfShares = shares.shape
    primer = np.array(get_prime_numbers(622))
    q =  primer[primer> shares[0].max()][0]
    result = np.zeros((m, n))
    for row in range(m):
        for col in range(n):
            y = shares[row, col, :]
            result[row, col] = lagrange(x, y, max_decrypt)% q
    return result



if __name__ == "__main__":

    args = parse_arguments()
    # [Encrypt]
    if args.encrypt:
        gray = cv2.imread(args.secret, cv2.IMREAD_GRAYSCALE)
        print("Construct Level 1 share.. ")
        # Step 1: (2, 2)-QSP based VCS of grayscale image
        QSP_shares = QSP_encrypt(gray)
        print("Construct Level 2 share.. ")
        # Step 2: Employing Shamir’s (k, m)-threshold scheme on each share
        for idx in range(len(QSP_shares)):
            level1_share = QSP_shares[idx]
            level1_share = np.round(level1_share*MAX_VALUE)

            # cv2.imwrite(os.path.join(args.output,  "share{}.png".format(idx+1)), level1_share)

            level2_shares = Shamir_encrypt(level1_share, args.k, args.m)
            print("Construct Codebook {} .. ".format(idx+1))
            # Step 3: Compression using Vector Quantization
            codebook = vector_quantization(level2_shares)
            
            # Save level 2 shares 
            for iidx in range(args.m):
                share = level2_shares[iidx]
                cv2.imwrite(os.path.join(args.output,  "share{}{}.png".format(idx+1, iidx+1)), share)
            # Save codebook
            saveCodebook(os.path.join(args.output,  "codebook{}.txt".format(idx+1)), codebook, args.m)

    # Decompress codebook into shares
    if args.decompress:
        for idx in range(2):
            codebook_path = os.path.join(args.shares, "codebook{}.txt".format(idx+1))
            codebook = read_codebook(codebook_path, args.m)
            shares = decompress_codebook(codebook, args.m)

            
            # Decrypt codebook 
            """
            for iidx in range(args.m):
                share = shares[iidx]
                cv2.imwrite(os.path.join(args.output,  "share{}{}_vq.png".format(idx+1, iidx+1)), share)
            """
    # [Decrypt] Regeneration of shares with random select shares
    if args.decrypt:

        for idx in range(2):
            assert args.k <= args.num_decrypt, "The number of decrypted shares needs to \
            be greater than the minimum k pieces required by Shamir's (k, m) scheme to reconstruct the secret."
            # np.random.seed(622)
            # Randomly choice shares
            select_list = np.random.choice(range(1, args.m+1), size=(args.num_decrypt), replace=False)
            print("Randomly select share images... ")
            for iidx in range(args.num_decrypt):
                share_name = "share{}{}.png".format(idx+1, select_list[iidx])
                print(share_name)
                level2_share = cv2.imread(os.path.join(args.shares, share_name),cv2.IMREAD_GRAYSCALE )
                if iidx == 0:
                    m, n = level2_share.shape
                    shares = np.zeros((m, n, args.num_decrypt))
                    shares[:, :, iidx] = level2_share
                else:
                    shares[:, :, iidx] = level2_share
            # Level 2 shares -> level 1 shares
            level1_share = Shamir_decrypt(shares, select_list, args.num_decrypt)

            if idx == 0:
                m, n = level1_share.shape
                level1_shares = np.zeros((m, n,  2)).astype(np.int64)

            level1_shares[:, : , idx] = level1_share
            

        # level 1 shares -> secret image
        recover_image = stacking(level1_shares).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output,  "decrypt_secret.png"), recover_image)
