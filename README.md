# C-SIP: Contrastive Spatial-Image Pre-training

[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)]()
[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Official implementation of Contrastive Learning of Image Representations Guided by Spatial Relations: [IEEE](https://ieeexplore.ieee.org/abstract/document/10943556).
[Open Access version](https://openaccess.thecvf.com/content/WACV2025/papers/Servant_Contrastive_Learning_of_Image_Representations_Guided_by_Spatial_Relations_WACV_2025_paper.pdf).

Published at the 25th IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

---

# Abstract

- State-of-the-Art spatially-aware models struggle with fine-grained spatial information because of the semantic training;
- We introduce C-SIP, Contrastive Spatial-Image Pre-training, a training strategy that leverages quantitative spatial relations for better spatially-aware image representations more in line with human perception;
- C-SIP models are experimentally evaluated on three downstream tasks: Spatial Relationship Recognition, Image Retrieval, and Visual Question Answering.

# Requirements

- Download the [COCO](https://cocodataset.org/#download) and [SpatialSense](https://github.com/princeton-vl/SpatialSense/tree/master) datasets for training the models.
- Download the [Unrel](https://github.com/jpeyre/unrel) and [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) for evaluating the models.
- Request the authors to provide the code snippet to compute [Force Banners](https://ieeexplore.ieee.org/document/9412316).

# Reference 

If you use C-SIP, please cite our paper:

```bibtex
@inproceedings{CSIP,
  author       = {Logan Servant and
                  Micha{\"{e}}l Cl{\'{e}}ment and
                  Laurent Wendling and
                  Camille Kurtz},
  title        = {Contrastive Learning of Image Representations Guided by Spatial Relations},
  booktitle    = {{IEEE/CVF} Winter Conference on Applications of Computer Vision, {WACV}
                  2025, Tucson, AZ, USA, February 26 - March 6, 2025},
  pages        = {2124--2133},
  publisher    = {{IEEE}},
  year         = {2025},
  doi          = {10.1109/WACV61041.2025.00213}
}
```
