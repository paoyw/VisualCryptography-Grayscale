# Visual Cryptography for Grayscale Image

## Methods
### [Extended Visual Cryptography for Natural Images](https://otik.uk.zcu.cz/handle/11025/5993)
- Encrypt
    ```bash
    python evc.py --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1 --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```

### [Halftone Visual Cryptography Embedding a Natural Grayscale Image Based on Error Diffusion Technique](https://ieeexplore.ieee.org/abstract/document/4285100/)
- Encrypt
    ```bash
    python ed.py --carry0 CARRY_IMAGE_0 --carry1 CARRY_IMAGE_1 --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```

### [Image Size-Preserving Visual Cryptography by Error Diffusion](https://catalog.lib.kyushu-u.ac.jp/opac_detail_md/?lang=0&amode=MD100000&bibid=1936207)
- Encrypt
    ```bash
    python isp_ed.py --secret SECRET_IMAGE --share0 SHARE_IMAGE_0 --share1 SHARE_IMAGE_1
    ```

## File structure
```
.
├── README.md
└── requirements.txt
```