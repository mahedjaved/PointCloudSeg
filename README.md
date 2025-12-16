# PointCloudSeg
Project for designing initial version of an ML pipeline for Point Cloud Semantic Segmentation that can grow into a advanced-grade ML pipeline. Focusing more on methodology, licensing, workflow, and reasoning from data acquisition to data preparation and a training the model.

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
- VoxelDownsampler : Reduces the density of the point cloud by downsampling voxels to a fixed grid size.
- PointCloudNormalizer : Normalizes the point cloud data by centering the origin and scaling to unit sphere

- Aiming to abstract the augmentation process into a generic framework that can be easily integrated into any existing deep learning pipeline for point cloud semantic segmentation.
- Using Python ABC module to create a base class for augmentation that can be inherited by any augmentation class.
- Main code inspired from: https://torch-points3d.readthedocs.io/en/latest/src/api/transforms.html - improvements include SOLID inspired design principles for better modularity and maintainability

# 2. Pre-Processing
- Species Mapping : Maps tree species names to integer class IDs for training.
- One-Hot Encoding : Converts textual labels (species names) into numeric labels suitable for ML models (onehot vector)
- Tree Extraction : Find all LiDAR points within a radius on XY plane and return their features (XYZ + intensity)
- Preprocessing Stage : 
    - Reads a .las file (3D LiDAR point cloud of a plot).
    - Converts the LAS points into a NumPy array.
    - Loops over each tree in the GeoDataFrame (gdf_plot) corresponding to that plot:
    - Checks species and geometry.
    - Extracts the tree’s points using extract_tree_points.
    - Creates a label array for the tree points.
    - Applies the point cloud preprocessor (normalization, voxelization, augmentation, sampling).
    - Saves the processed tree as a .npz file and store in data/processed/ folder
    - Returns the list of generated sample IDs as part of the preprocessing function 
- Main Preproc Runner:
    - Sets up directories for output (processed/samples).
    - Loads field survey data from a GeoJSON file containing tree positions and species.
    - Initializes the PointCloudPreprocessor and species mapping.
    - Loops through all .las files in the ALS dataset:
        - Extracts the plot ID from the file name.
        - Filters the GeoDataFrame to get trees in that plot.
        - Processes the plot using process_plot.
    - Perform post-processing split:
        - Slits the generated data samples into train (80%) and validation (20%).
        - Saves the train/val IDs as text files for later use in training.
- Code partly inspired from : https://www.kaggle.com/code/hedifeki/tree-detection - edits involve having a runner class for managing preprocessing and training tasks

# 3. Train
- Implements a PointNet-style neural network for 3D point cloud classification
- Uses PyTorch
- Trains and evaluates the model on a tree point-cloud dataset
- Code inspired from : https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_cls.py - modifications include minimal adaptation for tree point-cloud classification

Architecture is inspired by PointNet++, but simplified (no sampling/grouping)
# References
[1] Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey
Qinfeng Zhua,b, Lei Fana, 1
, Ningxin Wenga, : https://arxiv.org/pdf/2308.12113

N.B
“The pip show command does not display license metadata for setuptools. The package is maintained by the Python Packaging Authority and licensed under MIT, as documented in its official repository and PyPI listing. All Python dependencies were audited for licensing. All required libraries use permissive licenses (MIT, BSD, Apache, PSF). Where license metadata was missing (e.g., setuptools), the license was manually verified from the official project repository.”