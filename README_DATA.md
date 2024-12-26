# CARE

Data used in our TIP paper:

> **Concept-Aware Video Captioning: Describing Videos With Effective Prior Information**
>
> Bang Yang, Meng Cao and Yuexian Zou*.
>
> [[IEEE Xplore](https://ieeexplore.ieee.org/document/10233200)]


## Pre-Processed Data

**You can download our preprocessed data from [Google Drive](https://drive.google.com/drive/folders/1v6CFTrh4SWCZnoURT89u6DKQ2o5P_uys?usp=sharing) or [OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/2101112290_pkueducn_onmicrosoft_com/EknFBr4uhGpDux8dqCPlD5sB5C_M4Bw6VO73e-KX8uijsA?e=uktfnc), which follows the structure below:**


```
└── base_data_path
    ├── MSRVTT
    │   ├── feats
    │   │   ├── image_R101_fixed60.hdf5
    │   │   ├── ...
    │   │   ├── CLIP_ViT-B-32.hdf5
    │   │   ├── motion_resnext101_kinetics_fixed60.hdf5
    │   │   └── audio_vggish_audioset_fixed60.hdf5
    │   ├── retrieval
    │   │   ├── ...
    │   │   └── CLIP_ViT-B-32_unique.hdf5
    │   ├── info_corpus.pkl 
    │   └── refs.pkl
    ├── MSVD
    │   ├── feats
    │   │   └── ...
    │   ├── retrieval
    │   │   └── ...
    │   ├── info_corpus.pkl
    │   └── refs.pkl
    └── VATEX
        ├── feats
        │   └── ...
        ├── retrieval
        │   └── ...
        ├── info_corpus.pkl
        └── refs.pkl
```
Please remember to modify `base_data_path` in [config/Constants.py](config/Constants.py)

## Raw Videos

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Official Link</th><th>Shared Link (Ours)</th>
    </tr>
    <tr align="center">
        <td>MSVD</td><td><a href="https://www.cs.utexas.edu/users/ml/clamp/videoDescription/">Link</a></td><td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/2101112290_pkueducn_onmicrosoft_com/ESi2AhDhuMpPsfv5E3N9xtsBhbraiiC4ZuAhwCdNS7kGYA?e=LPcfkl">Onedrive</a>, <a href="https://disk.pku.edu.cn/link/AA9E57BB3055344D98BCED580891278655">PKU Yun</a> (1.7G)</td>
    </tr>
    <tr align="center">
        <td>MSRVTT</td><td><a href="http://ms-multimedia-challenge.com/2016/dataset">Link (expired)</a></td><td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/2101112290_pkueducn_onmicrosoft_com/EW8dnlrbXrhPpHCzqUWYBmEBy_15l4nQuZBuIS2akdIWwg?e=mxCEwZ">Onedrive</a>, <a href="https://disk.pku.edu.cn/link/AACF5DF7B019D64AA7A956E99A5A3201ED">PKU Yun</a> (6.1G)</td>
    </tr>
    <tr align="center">
        <td>VATEX</td><td><a href="https://eric-xw.github.io/vatex-website/download.html">Link</a></td><td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/2101112290_pkueducn_onmicrosoft_com/EbznKwMvV-1FsxxxRvbiu1cB5aC-NTspM1y5zkyJq6rZSQ?e=IcpHpT">Onedrive</a>, <a href="https://disk.pku.edu.cn/link/AA70A2F20E92AA48F48153C82119347504">PKU Yun</a> (37.3G); <a href="https://pan.baidu.com/s/1Vr1ppSCmKBNfMNfSf0w2rA">Baidu Yun</a> (extract code: t3ge)</td>
    </tr>
</table>
</div>

**You can download raw videos from our shared links. Please organize them as follows:**

```
└── base_data_path
    ├── MSVD
    │   └── all_videos
    │       ├── video0.avi
    │       ├── ...
    │       └── video1969.avi
    ├── MSRVTT
    │   └── all_videos
    │       ├── video0.mp4
    │       ├── ...
    │       └── video9999.mp4
    └── VATEX
         └── all_videos
             ├── video0.mp4
             ├── ...
             └── video34990.mp4
```

**Note:** 
- The original names of MSVD and VATEX videos do not follow the format `videoXXX`, we provide the mappings in [data/msvd_mapping.txt](data/msvd_mapping.txt) and [data/vatex_mapping.txt](data/vatex_mapping.txt).
- Considering the difficulties to download raw VATEX videos, we share them!
- The official train/val/test splits of VATEX is 25,991: 3,000: 6,000. However, some video clips are no longer available, resulting in the splits 25,006: 2,893: 5,792 (in our case). The same splits are used in our other papers, check them out if you are interested ✨: 

    > [**MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning**](https://aclanthology.org/2023.acl-long.664/)<br>
    > Accepted by ACL 2023 | [[Code]](https://github.com/yangbang18/MultiCapCLIP)<br>
    > Bang Yang, Fenglin Liu, Xian Wu, Yaowei Wang, Xu Sun, and Yuexian Zou

    > [**CLIP Meets Video Captioning: Concept-Aware Representation Learning Does Matter**](https://arxiv.org/abs/2111.15162)<br>
    > Accepted by PRCV 2022 | [[Code]](https://github.com/yangbang18/CLIP-Captioner)<br>
    > Bang Yang, Tong Zhang and Yuexian Zou


## Prepare data on your own

**1. Download Raw Videos**

**2. Feature Extraction (Image, Motion, Audio)**

* Extract video frames

  ```
  MSRVTT_ROOT=$base_data_path/MSRVTT

  python pretreatment/extract_frames_from_videos.py \
  --video_path $MSRVTT_ROOT/all_videos \
  --frame_path $MSRVTT_ROOT/all_frames \
  --video_suffix mp4 \
  --frame_suffix jpg \
  --strategy 0
  ```
  
* Extract image features from ImageNet pre-trained models

  ```
  MSRVTT_ROOT=$base_data_path/MSRVTT

  python pretreatment/extract_image_feats_from_frames.py \
  --frame_path $MSRVTT_ROOT/all_frames \
  --feat_path $MSRVTT_ROOT/feats \
  --feat_name image_R101_fixed60.hdf5 \
  --model resnet101 \
  --frame_suffix jpg \
  --gpu 0

  python pretreatment/extract_image_feats_from_frames.py \
  --frame_path $MSRVTT_ROOT/all_frames \
  --feat_path $MSRVTT_ROOT/feats \
  --feat_name image_IRv2_fixed60.hdf5 \
  --model inceptionresnetv2 \
  --frame_suffix jpg \
  --gpu 0
  ```
* Extract image features from CLIP models
  ```
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN50
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN101
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN50x4
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN50x16
  python pretreatment/clip_feats.py --dataset MSRVTT --arch ViT-B/32
  python pretreatment/clip_feats.py --dataset MSRVTT --arch ViT-B/16

  # after you finetuning CLIP on MSRVTT (e.g., with open_clip), you can extract image features like this:
  python pretreatment/clip_feats.py --dataset MSRVTT --arch ViT-B/32 --checkpoint_path /path/to/finetuned_checkpoint --postfix ft
  ```
* Extracting motion features: refer to [yangbang18/video-classification-3d-cnn](https://github.com/yangbang18/video-classification-3d-cnn)

* Extracting audio features: refer to [yangbang18/vggish](https://github.com/yangbang18/vggish)

**3. Feature Extraction (Captions)**

This is a premininary step for step 4. 

```
python pretreatment/clip_text_embs.py --dataset MSRVTT --arch RN50
python pretreatment/clip_text_embs.py --dataset MSRVTT --arch RN50x16
python pretreatment/clip_text_embs.py --dataset MSRVTT --arch ViT-B/32
```

**4. Feature Extraction (Retrieval)**

**Note:** 
- CLIP's image features (step 2) and text features (step 3) must be done before this step. 
- In our TIP paper, we always use CLIP's ViT-B/32 to obtain text features of captions.

```
# in this case
# ViT-B/32 is used for retrieval (step 2 image features + step 3 text features)
# ViT-B/32 is used for feature extraction (step 3 text features)

python pretreatment/clip_retrieval.py \
--dataset MSRVTT \
--arch ViT-B/32 \
--n_frames 28 \
--topk 100 \
--unique
```

```
# in this case
# ViT-B/16 is used for retrieval (step 2 image features + step 3 text features)
# ViT-B/32 is used for feature extraction (step 3 text features)

python pretreatment/clip_retrieval.py \
--dataset MSRVTT \
--arch ViT-B/16 \
--arch_f ViT-B/32 \
--n_frames 28 \
--topk 100 \
--unique
```

```
# in this case
# ViT-B/32 is used for retrieval
# ViT-B/32 is used for feature extraction
# only 0.1% training captions are used as the retrieval database
python pretreatment/clip_retrieval.py \
--dataset MSRVTT \
--arch ViT-B/32 \
--n_frames 28 \
--unique \
--ratio 0.1 \
--topk 20
```



**5. Preprocess Corpora**
  ```
  python pretreatment/prepare_corpora.py --dataset MSVD --sort_vocab --attribute_first
  python pretreatment/prepare_corpora.py --dataset MSRVTT --sort_vocab --attribute_first
  python pretreatment/prepare_corpora.py --dataset VATEX --sort_vocab --attribute_first
  ```
**Note:** 
- When processing MSVD's annotations, we directly use the off-the-shelf `refs.pkl` rather than preprocess it from scratch due to the expired official link.
- When processing MSRVTT's annotations, our code will try to download `videodatainfo.json` from the official link, which, however, seems to be expired. You can download this file from our data link.
- When processing VATEX's annotations, our code will determine the actual train/val/test splits according to the available raw videos. This is because some videos can not be access anymore.


## Citation

Please **[★star]** this repo and **[cite]** the following papers if you feel our code and data useful to your research:

```
@ARTICLE{yang2023CARE,
  author={Yang, Bang and Cao, Meng and Zou, Yuexian},
  journal={IEEE Transactions on Image Processing}, 
  title={Concept-Aware Video Captioning: Describing Videos With Effective Prior Information}, 
  year={2023},
  volume={32},
  number={},
  pages={5366-5378},
  doi={10.1109/TIP.2023.3307969}
}
```
