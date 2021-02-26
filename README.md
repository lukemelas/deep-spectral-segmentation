<div align="center">    
 
## Project Name 

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
### Description   
<!-- TODO: Add abstract -->

### How to run   

#### Dependencies
<!-- TODO: Add description -->
 - PyTorch (tested on version 1.7.1, but should work on any version)
 - Hydra: `pip install hydra-core --pre`
 - Other: `pip install albumentations tqdm tensorboard`
 - PyTorch Image Models: `pip install timm`
 - WandB (optional): `pip install wandb`

#### Training
<!-- TODO: Add description -->
```bash
python main.py 
```


#### Evaluation
<!-- TODO: Add description -->
```bash
python main.py eval=True
```

#### Debugging
<!-- TODO: Add description -->
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config-name="debug.yaml" data.train.root="/home/luke/machine-learning-datasets/image-classification/imagenet/val" data.val.root="/home/luke/machine-learning-datasets/image-classification/imagenet/val" data.loader.batch_size=256 data.loader.num_workers=16 data.loader.pin_memory=True optimizer.lr=1e-3
```

Example run [here](https://wandb.ai/lukemelas2/template/runs/1huwqeth).


#### Pretrained models


#### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
