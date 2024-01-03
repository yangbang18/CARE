# CARE

PyTorch Implementation of our TIP paper:

> **Concept-Aware Video Captioning: Describing Videos With Effective Prior Information**
>
> Bang Yang, Meng Cao and Yuexian Zou.
>
> [[IEEE Xplore](https://ieeexplore.ieee.org/document/10233200)]

## TOC

- [CARE](#care)
  - [TOC](#toc)
  - [Update Notes](#update-notes)
  - [Environment](#environment)
  - [Running](#running)
    - [Overview](#overview)
    - [Training](#training)
    - [Testing](#testing)
    - [Show Results](#show-results)
  - [Reproducibility](#reproducibility)
    - [Main Experiments](#main-experiments)
    - [Ablation Study](#ablation-study)
    - [Analysis](#analysis)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Update Notes
**[2023-10-22]** We release the code and data.


## Environment
Clone and enter the repo:

```shell
git clone https://github.com/yangbang18/CARE.git
cd CARE
```

We has refactored the code and tested it on: 
- `Python` 3.9
- `torch` 1.13.1
- `cuda` 11.7 

Other versions are also work, e.g., `Python` 3.7, `torch` 1.7.1, and `cuda` 10.1. 

Please change the version of torch and cuda according to your hardwares.

```shell
conda create -n CARE python==3.9
conda activate CARE

# Install a proper version of torch, e.g.:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117  -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Note that `torch < 1.7.1` is imcompatible with the package `clip`
pip install -r requirement.txt
```


## Running

### Overview
**1. Supported datasets** (please follow [README_DATA.md](/README_DATA.md) to prepare data)
- `MSVD`
- `MSRVTT`
- `VATEX`

**2. Supported methods**, whose configurations can be found in [config/methods.yaml](config/methods.yaml)

- `Transformer`: our baseline (autoregressive)
- `TopDown`: same encoder as `Transformer`, a two layer LSTM decoder (autoregressive)
- `ARB`: a slight different encoder compared with `Transformer` (autoregressive)
- `NACF`: same encoder as `ARB`, different decoding algorithm (non-autoregressive)
- ...

**3. Supported feats**, whose configurations can be found in [config/feats.yaml](config/feats.yaml)

**4. Supported modality combinations**: any combination of `a` (audio), `m` (motion) and `i` (image).

**5. Supported architectures**, whose configurations can be found in [config/archs.yaml](config/archs.yaml)

- `base`: used in MSVD, MSRVTT (model dimension: `512`)
- `median`: used in VATEX (model dimension: `768`)
- `large`: used in VATEX (model dimension: `1,024`)

**6. Supported tasks**, whose configurations can be found in [config/tasks.yaml](config/tasks.yaml).

- `Base`: take different combination of features for decoding.
- `CABase`: visual-driven concept detection and ``Cross -> Semantic'' local semantic guidance.
- `CARE`: multimodal-driven concept detection (MCD) and global-local semantic guidance (G-LSG).
- `Concept`: basic implementations of `CABase` and `CARE`, it supports the modification of hyper-parameters related to concept detection.

### Training
**Command Format:**

For the `Base` task:
```
python train.py \
--dataset $dataset_name \
--method $method_name \
--feats $feats_name \
--modality $modality_combination \
--arch $arch_name \
--task $task_name
```

For `CAbase`, `CARE`, and `Concept` tasks:
```
python train.py \
--dataset $dataset_name \
--method $method_name \
--feats $feats_name \
--decoder_modality_flags $flags1 \
--predictor_modality_flags $flags2 \
--arch $arch_name \
--task $task_name
```
**Note:** the only different is that we need to specify `decoder_modality_flags` and `predictor_modality_flags` rather than `modality`. This is because modalities used for concept detection and decoding can be different. Here are some mappings between `flag` and `modality` (refer to the `flag2modality` of [config/Constants.py](config/Constants.py)):
  - `I` (image): i
  - `V` (vision): mi
  - `A` (audio): a
  - `VA` (vision + audio): ami
  - `VAT` (vision + audio + text): amir

**Example:**

```shell
python train.py \
--dataset MSRVTT \
--method Transformer \
--arch base \
--task Base \
--feats ViT \
--modality ami  

python train.py \
--dataset MSRVTT \
--method Transformer \
--arch base \
--task CARE \
--feats ViT \
--decoder_modality_flags VA \
--predictor_modality_flags VAT
```

### Testing

```shell
ckpt=/path/to/checkpoint

python translate.py --checkpoint_paths $ckpt

# evaluate on the validation set
python translate.py --checkpoint_paths $ckpt --mode validate

# evaluate on the validation set & save results to a csv file (same directory as the checkpoint)
python translate.py --checkpoint_paths $ckpt --mode validate --save_csv --csv_name val_result.csv

# evaluate on the validation set & save results to a csv file
python translate.py --checkpoint_paths $ckpt --mode validate --save_csv --csv_path ./results/csv --csv_name dummy.csv

# save caption predictions
python translate.py --checkpoint_paths $ckpt --json_path ./results/predictions --json_name dummy.json

# save detailed per-sample scores
python translate.py --checkpoint_paths $ckpt --save_detailed_scores_path ./results/detailed_scores/dummy.json
```

### Show Results
You can run the following command to gather results, where mean metric scores with their standard deviation across a number of runs are reported.

```
python misc/merge_csv.py --dataset MSVD --average --output_path ./results --output_name MSVD.csv
```

## Reproducibility
### Main Experiments (Compared with SOTA)
```
bash scripts/exp_main_MSVD.sh
bash scripts/exp_main_MSRVTT.sh
bash scripts/exp_main_VATEX.sh
```

### Ablation Study
**Note:** each script is **self-enclosed**, i.e., a model variant may be included in `N` differnt scripts, each of which will train the model in `K` (K=5 by default) different seeds (`[0, K)`), resulting in `N x K` runs. Just be careful.


- Main Ablation Study
  ```shell
  bash scripts/exp_ablation_main.sh
  ```

- Ablation on GSG and LSG variants
  ```shell
  bash scripts/exp_ablation_GLSG.sh
  ```

- Versatility of CARE on Various Encoder-Decoder Networks
  ```shell
  bash scripts/exp_versatility_of_CARE.sh
  ```
  **Note:** To reproduce SwinBERT, we do not train the model in an end-to-end manner. Instead, we conduct experiments on features extracted from fine-tuned SwinBERT checkpoints. Therefore, if you want to follow our implementations, please refer to [yangbang18/Video-Swin-Transformer](https://github.com/yangbang18/Video-Swin-Transformer) for feature extraction first (we do not share the feature files due to their large size). Otherwise, you should comment the command lines about SwinBERT in `scripts/exp_versatility_of_CARE.sh`.

- Measuring retrieval performance
  ```shell
  python pretreatment/clip_retrieval.py --dataset MSRVTT --n_frames 28 --eval --arch RN50
  python pretreatment/clip_retrieval.py --dataset MSRVTT --n_frames 28 --eval --arch RN50x16
  python pretreatment/clip_retrieval.py --dataset MSRVTT --n_frames 28 --eval --arch ViT-B/32

  python pretreatment/clip_retrieval.py --dataset MSRVTT --n_frames 28 --eval --arch ViT-B/32 --postfix ft
  python pretreatment/clip_retrieval.py --dataset MSRVTT --n_frames 28 --eval --arch ViT-B/16 --postfix ft
  ```
  **Note:** 
  - Image features of the first 3 commands are provided in our data, but the corresponding text features should be extracted by yourselves. Please refer to the step 3 in [README_DATA.md](README_DATA.md)
  - Image and text features of the bottom 2 commands are provided in our data, and they are extracted from CLIP fine-tuned on MSRVTT. Our fine-tuning process is done with [open_clip](https://github.com/mlfoundations/open_clip). 

### Analysis
Please refer to [notebooks](notebooks).


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

## Acknowledgement

- This codebase is built upon our previous one, i.e., [CLIP-Captioner](https://github.com/yangbang18/CLIP-Captioner).
