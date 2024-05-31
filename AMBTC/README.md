# AMBTC-based VC

We implement the paper [Reversible AMBTC-based secret sharing scheme with abilities of two decryptions](https://www.sciencedirect.com/science/article/pii/S1047320313002320). 

## Image setting
1.  **Secret Image:** Grayscale Image
2.  **Carry Image:** Binary Image, which size of the carrier image should be a multiple of the block size of the secret image.
3.  **Block Size:** 4 (If you want to use a different block size, you can modify the variable BLOCKSIZE in line 18)

## How to run? 

- Encrypt
    ```bash
    python3 AMBTC.py --secret ./path/to/secret/image --carry ./path/to/carry/image --output ./path/to/output/directory --encrypt
    ```
- Recontruct
    ```bash
    python3 AMBTC.py --shares ./path/to/shares/directory --output ./path/to/output/directory --reconstruct
    ```
- Decrypt 
    ```bash
    python3 AMBTC.py --shares ./path/to/shares/directory --output ./path/to/output/directory --decrypt
    ```

