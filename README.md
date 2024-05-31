# Visual Cryptography for Grayscale Image

## Methods
### [Extended Visual Cryptography for Natural Images](https://otik.uk.zcu.cz/handle/11025/5993)
- Encrypt
    ```bash
    python evc.py --carry0 CARRY_IMAGE_0 --carry1 CARRY_IMAGE_1 --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```
- Decrypt
    ```bash
    python evc.py --share0 SHARE_IMAGE_2 --share1 SHARE_IMAGE_1
    ```

### [Halftone Visual Cryptography Embedding a Natural Grayscale Image Based on Error Diffusion Technique](https://ieeexplore.ieee.org/abstract/document/4285100/)
- Encrypt
    ```bash
    python3 ed.py --carry0 CARRY_IMAGE_0 --carry1 CARRY_IMAGE_1 --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```

### [Image Size-Preserving Visual Cryptography by Error Diffusion](https://catalog.lib.kyushu-u.ac.jp/opac_detail_md/?lang=0&amode=MD100000&bibid=1936207)
- Encrypt
    ```bash
    python3 isp_ed.py --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```
- Decrypt
    ```bash
    python3 isp_ed.py --dec-only --dec DECRYPT_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```
### [Reversible AMBTC-based secret sharing scheme with abilities of two decryptions](https://www.sciencedirect.com/science/article/pii/S1047320313002320)
Please ensure the size of carry image and carry image. The size of the carry image needs to be a multiple of the block size of the secret image size. For example, if the size of carry image is (512, 512) and block size is 4, the secret image should be (128, 128) binary image. For easy use, we provide a binary NTU_emblem.png in the datas folder, which has a size of (128, 128). 

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
    

## File structure
```
.
├── datas
│   ├── README.md
│   ├── NTU_emblem.png
│   ├── aerials
│   │   ├── 2.1.01.tiff
│   │   └── ... 
│   ├── misc
│   │   ├── 4.1.01.tiff
│   │   └── ...
│   └── textures
│       ├── 1.1.01.tiff
│       └── ...
├── AMBTC
│   ├── README.md
│   ├── RUN_AMBTC.sh
│   └── AMBTC.py
|
├── ed.py
├── evc.py
├── isp_ed.py
├── README.md
├── requirements.txt
└── utils
    ├── halftone.py
    └── __init__.py
```
