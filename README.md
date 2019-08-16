# Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations
<p align="center"><img src="outline.jpg" alt="outline" width="90%"></p>
The code of:

Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, Jiwoon Ahn, Sunghyun Cho, and Suha Kwak, CVPR 2019 [[Paper]](https://arxiv.org/abs/1904.05044)

This repository contains a framework for learning instance segmentation with image-level class labels as supervision. The key component of our approach is Inter-pixel Relation Network (IRNet) that estimates two types of information: a displacement vector field and a class boundary map, both of which are in turn used to generate pseudo instance masks from CAMs.

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@InProceedings{Ahn_2019_CVPR,
author = {Ahn, Jiwoon and Cho, Sunghyun and Kwak, Suha},
title = {Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Prerequisite
* Python 3.7, PyTorch 1.1.0, and more in requirements.txt
* PASCAL VOC 2012 devkit
* NVIDIA GPU with more than 1024MB of memory

## Usage

#### Install python dependencies
```
pip install -r requirements.txt
```
#### Download PASCAL VOC 2012 devkit
* Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

#### Run run_sample.py or make your own script
```
python run_sample.py
```
* You can either mannually edit the file, or specify commandline arguments.

#### Train Mask R-CNN or DeepLab with the generated pseudo labels
* For the reports, we used [Detectron](https://github.com/facebookresearch/Detectron).
  * Run step/make_cocoann.py to create COCO-style annotations.
  * Note: Do not employ https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip to measure the performance of the Mask R-CNN! It only contains bounding box annotations.
* [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) now supports Mask R-CNN and DeepLab. I personally recommend to use this.

## TO DO
* Training code for MS-COCO
* Code refactoring
* IRNet v2
