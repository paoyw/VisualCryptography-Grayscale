# AMBTC-based VC

We implement the paper [A Hierarchical Image Cryptosystem Based on Visual Cryptography and Vector Quantization](https://link.springer.com/chapter/10.1007/978-981-13-1540-4_1). 

## setting
1.  **Secret Image:** Grayscale Image
2. **Shamir's (k, m) threshold scheme**: m is the total number of shares whereas k is the minimum
number of shares required to recover the secret.
3. **num_decrypt:** The number of shares be decrypted
## How to run? 
Here we are using Shamir's (3, 6) threshold scheme.

- **Encrypt:** Encrypt secret image.
    ```bash
    python3 HVC.py --secret ./path/to/secret/image --output ./path/to/output/directory --k 3 --m 6 --encrypt 
    ```
- **Decompress:** Decompress codebook.
    ```bash
    python3 HVC.py --shares ./path/to/codebook/directory --output ./path/to/output/directory --k 3 --m 6 --decompress 
    ```
- **Decrypt:** Decrypt Carry Image.
    ```bash
    python3 HVC.py --shares ./path/to/shares/directory --output ./path/to/output/directory --k 3 --m 6  --num_decrypt 5 --decrypt
    ```
    

