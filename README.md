<div align="center">
 
## Deep Spectral Methods for Unsupervised Localization and Segmentation (CVPR 2022 - Oral)

[![Project](http://img.shields.io/badge/Project%20Page-3d3d8f.svg)](https://lukemelas.github.io/deep-spectral-segmentation/)
[![Demo](http://img.shields.io/badge/Demo-9acbff.svg)](https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation)
[![Conference](http://img.shields.io/badge/CVPR-2022-4b44ce.svg)](#)
[![Paper](http://img.shields.io/badge/Paper-arxiv.1001.2234-B31B1B.svg)](#)

</div>

### Description
This code accompanies the paper [Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization](https://lukemelas.github.io/deep-spectral-segmentation/). 

### Abstract

Unsupervised localization and segmentation are long-standing computer vision challenges that involve decomposing an image into semantically-meaningful segments without any labeled data. These tasks are particularly interesting in an unsupervised setting due to the difficulty and cost of obtaining dense image annotations, but existing unsupervised approaches struggle with complex scenes containing multiple objects. Differently from existing methods, which are purely based on deep learning, we take inspiration from traditional spectral segmentation methods by reframing image decomposition as a graph partitioning problem. Specifically, we examine the eigenvectors of the Laplacian of a feature affinity matrix from self-supervised networks. We find that these eigenvectors already decompose an image into meaningful segments, and can be readily used to localize objects in a scene. Furthermore, by clustering the features associated with these segments across a dataset, we can obtain well-delineated, nameable regions, i.e. semantic segmentations. Experiments on complex datasets (Pascal VOC, MS-COCO) demonstrate that our simple spectral method outperforms the state-of-the-art in unsupervised localization and segmentation by a significant margin. Furthermore, our method can be readily used for a variety of complex image editing tasks, such as background removal and compositing.

### Demo
Please check out our interactive demo on [Huggingface Spaces](https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation)! The demo enables you to upload an image and outputs the eigenvectors extracted by our method. It does not perform the downstream tasks in our paper (e.g. semantic segmentation), but it should give you some intuition for how you might use utilize our method for your own research/use-case. 

### Examples

![Examples](https://lukemelas.github.io/deep-spectral-segmentation/images/example.png)

### How to run   

#### Dependencies
The minimal set of dependencies is listed in `requirements.txt`.

#### Data Preparation

The data preparation process simply consists of collecting your images into a single folder. Here, we describe the process for [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012//). Pascal VOC 2007 and MS-COCO are similar. 

Download the images into a single folder. Then create a text file where each line contains the name of an image file. For example, here is our initial data layout:
```
data
└── VOC2012
    ├── images
    │   └── {image_id}.jpg
    └── lists
        └── images.txt
```

#### Extraction

We first extract features from images and stores these into files. We then extract eigenvectors from these features. Once we have the eigenvectors, we can perform downstream tasks such as object segmentation and object localization. 

The primary script for this extraction process is `extract.py` in the `extract/` directory. All functions in `extract.py` have helpful docstrings with example usage. 

##### Step 1: Feature Extraction

First, we extract features from our images and save them to `.pth` files. 

With regard to models, our repository currently only supports DINO, but other models are easy to add (see the `get_model` function in `extract_utils.py`). The DINO model is downloaded automatically using `torch.hub`. 

Here is an example using `dino_vits16`:

```bash
python extract.py extract_features \
    --images_list "./data/VOC2012/lists/images.txt" \
    --images_root "./data/VOC2012/images" \
    --output_dir "./data/VOC2012/features/dino_vits16" \
    --model_name dino_vits16 \
    --batch_size 1
```

##### Step 2: Eigenvector Computation

Second, we extract eigenvectors from our features and save them to `.pth` files. 

Here, we extract the top `K=5` eigenvectors of the Laplacian matrix of our features:

```bash
python extract.py extract_eigs \
    --images_root "./data/VOC2012/images" \
    --features_dir "./data/VOC2012/features/dino_vits16" \
    --which_matrix "laplacian" \
    --output_dir "./data/VOC2012/eigs/laplacian" \
    --K 5
```

The final data structure after extracting eigenvectors looks like:
```
data
├── VOC2012
│   ├── eigs
│   │   └── {outpur_dir_name}
│   │       └── {image_id}.pth
│   ├── features
│   │   └── {model_name}
│   │       └── {image_id}.pth
│   ├── images
│   │   └── {image_id}.jpg
│   └── lists
│       └── images.txt
└── VOC2007
    └── ...
```

At this point, you are ready to use the eigenvectors for downstream tasks such as object localization, object segmentation, and semantic segmentation. 

#### Object Localization

First, clone the `dino` repo inside this project root (or symlink it).
```bash
git clone https://github.com/facebookresearch/dino
```

Run the steps above to save your eigenvectors inside a directory, which we will now call `${EIGS_DIR}`. You can then move to the `object-localization` directory and evaluate object localization with:
```bash
python main.py \
    --eigenseg \
    --precomputed_eigs_dir ${EIGS_DIR} \
    --dataset VOC12 \
    --name "example_eigs"
```

#### Object Segmentation

To perform object segmentation (i.e. single-region segmentations), you first extract features and eigenvectors (as described above). You then extract coarse (i.e. patch-level) single-region segmentations from the eigenvectors, and then turn these into high-resolution segmentations using a CRF.

Below, we will give example commands for the CUB bird dataset (`CUB_200_2011`). To download this dataset, as well as the three other object segmentation datasets used in our paper, you can follow the instructions in [unsupervised-image-segmentation](https://github.com/lukemelas/unsupervised-image-segmentation). Then make sure to specify the `data_root` parameter in the `config/eval.yaml`.

For example:
```bash

# Example dataset
DATASET=CUB_200_2011

# Features
python extract.py extract_features \
    --images_list "./data/object-segmentation/${DATASET}/lists/images.txt" \
    --images_root "./data/object-segmentation/${DATASET}/images" \
    --output_dir "./data/object-segmentation/${DATASET}/features/dino_vits16" \
    --model_name dino_vits16 \
    --batch_size 1

# Eigenvectors
python extract.py extract_eigs \
    --images_root "./data/object-segmentation/${DATASET}/images" \
    --features_dir "./data/object-segmentation/${DATASET}/features/dino_vits16/" \
    --which_matrix "laplacian" \
    --output_dir "./data/object-segmentation/${DATASET}/eigs/laplacian_dino_vits16" \
    --K 2 \


# Extract single-region segmentatiosn
python extract.py extract_single_region_segmentations \
    --features_dir "./data/object-segmentation/${DATASET}/features/dino_vits16" \
    --eigs_dir "./data/object-segmentation/${DATASET}/eigs/laplacian_dino_vits16" \
    --output_dir "./data/object-segmentation/${DATASET}/single_region_segmentation/patches/laplacian_dino_vits16"

# With CRF
# Optionally, you can also use `--multiprocessing 64` to speed up computation by running on 64 processes
python extract.py extract_crf_segmentations \
    --images_list "./data/object-segmentation/${DATASET}/lists/images.txt" \
    --images_root "./data/object-segmentation/${DATASET}/images" \
    --segmentations_dir "./data/object-segmentation/${DATASET}/single_region_segmentation/patches/laplacian_dino_vits16" \
    --output_dir "./data/object-segmentation/${DATASET}/single_region_segmentation/crf/laplacian_dino_vits16" \
    --downsample_factor 16 \
    --num_classes 2
```

After this extraction process, you should have a file with full-resolution segmentations. Then to evaluate on object segmentation, you can move into the `object-segmentation` directory and run `python main.py`. For example:

```bash
python main.py predictions.root="./data/object-segmentation" predictions.run="single_region_segmentation/crf/laplacian_dino_vits16"
```

By default, this assumes that all four object segmentations are available. To run on a custom dataset or only a subset of these datasets, simply edit `configs/eval.yaml`. 

Also, if you want to visualize your segmentations, you should be able to use `streamlit run extract.py vis_segmentations` (after installing streamlit). 

#### Semantic Segmentation

For semantic segmentation, we provide full instructions in the `semantic-segmentation` subfolder.

#### Acknowledgements

L. M. K. acknowledges the generous support of the Rhodes Trust. C. R. is supported by Innovate UK (project 71653) on behalf of UK Research and Innovation (UKRI) and by the European Research Council (ERC) IDIU-638009. I. L. and A. V. are supported by the VisualAI EPSRC programme grant (EP/T028572/1).

We would like to acknowledge LOST ([paper](https://arxiv.org/abs/2109.14279) and [code](https://github.com/valeoai/LOST)), whose code we adapt for our object localization experiments. If you are interested in object localization, we suggest checking out their work! 

#### Citation   
```
@inproceedings{
    melaskyriazi2022deep,
    title={Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization}
    author={Luke Melas-Kyriazi and Christian Rupprecht and Iro Laina and Andrea Vedaldi}
    year={2022}
    booktitle={CVPR}
}
```   
