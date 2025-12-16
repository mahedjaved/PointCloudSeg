# PointCloudSeg
Project placeholder for designing initial version of an ML pipeline for Point Cloud Semantic Segmentation that can grow into a advanced-grade ML pipeline. Focusing more on methodology, licensing, workflow, and reasoning from data acquisition to data preparation and a training the model.

Dir Structure
```
PointCloudSeg/
├── configs/
│   └── default.yaml
├── data/
│   └── raw/
├── eda/
│   └── eda.py
├── models/
│   └── pointnet.py
├── preprocessing/
│   └── preprocess.py
├── utils/
│   └── utils.py
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

# 1. Augmentation
- Random Rotation : Rotate the point cloud around the z-axis by a random angle.
- Random Scaling : Scale the point cloud by a random factor.
- Random Jitter : Add random noise (Gaussian/normal) to the point cloud.
- Random Flip : Flip the point cloud along the x-axis with a probability of 0.5.

- Aiming to abstract the augmentation process into a generic framework that can be easily integrated into any existing deep learning pipeline for point cloud semantic segmentation.
- Using Python ABC module to create a base class for augmentation that can be inherited by any augmentation class.
- Main code inspired from: https://torch-points3d.readthedocs.io/en/latest/src/api/transforms.html

# References
[1] Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey
Qinfeng Zhua,b, Lei Fana, 1
, Ningxin Wenga, : https://arxiv.org/pdf/2308.12113

N.B
“The pip show command does not display license metadata for setuptools. The package is maintained by the Python Packaging Authority and licensed under MIT, as documented in its official repository and PyPI listing. All Python dependencies were audited for licensing. All required libraries use permissive licenses (MIT, BSD, Apache, PSF). Where license metadata was missing (e.g., setuptools), the license was manually verified from the official project repository.”