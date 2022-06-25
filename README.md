# TEAR: Semi-Supervised Medical Image Classification with Temporal Knowledge-Aware Regularization

This repository is an official PyTorch implementation of the paper "Semi-Supervised Medical Image Classification with Temporal Knowledge-Aware Regularization" from MICCAI 2022.

![image][https://github.com/QiushiYang/TEAR/edit/main/figs/TEAR.png]

We propose TEmporal knowledge-Aware Regularization (TEAR) for semi-supervised medical image classification. Instead of using hard pseudo labels to train models roughly, we design Adaptive Pseudo Labeling (AdaPL), a mild learning strategy that relaxes hard pseudo labels to soft-form ones and provides a cautious training. AdaPL is built on a novel theoretically derived loss estimator, which approximates the loss of unlabeled samples according to the temporal information across training iterations, to adaptively relax pseudo labels. To release the excessive dependency of biased pseudo labels, we take advantage of the temporal knowledge and propose Iterative Prototype Harmonizing (IPH) to encourage the model to learn discriminative representations in an unsupervised manner. The core principle of IPH is to maintain the harmonization of clustered prototypes across different iteration.

## Dependencies
All experiments use the PyTorch library. We recommend installing the following package versions:

* &nbsp;&nbsp; python=3.7 

* &nbsp;&nbsp; pytorch=1.6.0

* &nbsp;&nbsp; torchvision=0.7.0

Dependency packages can be installed using following command:
```
pip install -r requirements.txt
```

## Quickstart

* Training
```python
python Train_TEAR.py 
    --gpu=0 
    --dataset=isic2018 
    --n-classes=7 
    --backbone=alexnet 
    --n-labeled=350 
    --batchsize=128 
    --thr=0.95 
    --n-epoches=256 
    --setting=ISIC_350
```

## Acknowledge
* The implementation of baseline method is based on: [CoMatch](https://github.com/salesforce/CoMatch)

