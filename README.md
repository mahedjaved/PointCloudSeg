# PointCloudSeg
This project is a POC and a WIP for designing initial stages of an ML pipeline for Point Cloud Semantic Segmentation that can grow into a advanced-grade ML pipeline. Focusing more on methodology, licensing, workflow, and reasoning from data acquisition to data preparation and a training the model. The idea is to train a model that can capture statistical correlations between various species of trees captured in a forest environment. The patterns and trends learned will direclty impact downstream applications that will operate on generating a 3D forest like environmnet for stealth action video

# Project Structure
```
PointCloudSeg/
├── configs/           # Stores configuration files (e.g., default.yaml) for experiments and training setups
├── data/              # Contains raw and processed datasets
│   └── raw/           # Holds unprocessed LiDAR/ALS point cloud data
│   └── processed/     # Holds the processed counterpart to raw
├── eda/               # Scripts for exploratory data analysis and visualizations
│   └── eda.py         # Performs EDA on the dataset (plots, stats, correlations)
│   └── insight.py     # Summarizes key findings and trends from the EDA
├── preprocessor/      # Contains code for preprocessing and augmenting point cloud data
│   └── datapreprocessor.py  # Handles tree point extraction, normalization, voxelization
│   └── augmentation.py      # Implements modular point cloud augmentation transforms
├── train/             # Training pipeline for the PointNet-style model
│   └── trainer.py     # Defines training loops, evaluation, and checkpointing
├── utils/             # Utility functions for file handling, data loading, and helpers
│   └── utils.py       # Contains reusable helper functions
├── config.py          # Loads and parses configuration files for experiments
├── .gitignore         # Specifies files/folders to exclude from Git version control
├── requirements.txt   # Lists Python dependencies required to run the project
├── LIBLICENSES.md     # Records licensing information for all used libraries
├── README.md          # Project overview, structure, methodology, and references
└── MODEL_CHOICES.md   # Summarizes different model architectures considered for the project

```

# 1. Augmentation

This folder outlines a modular point-cloud augmentation framework designed to improve model robustness and generalisation. It applies geometric and noise-based transformations to LiDAR point clouds while preserving semantic structure, and is implemented using an extensible, SOLID-inspired design that can be reused across different point-cloud learning pipelines.
 The augmentation framework includes the following transformations:
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

This folder contains lines the data preprocessing pipeline that converts raw ALS LiDAR data and field survey annotations into per-tree training samples. It covers species label encoding, tree-level point extraction, normalization, voxelization, augmentation, and dataset splitting, producing standardized .npz samples suitable for efficient model training.

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

This folder contains logic for the training pipeline for a PointNet-style neural network used for tree-level point-cloud segmentation. It details how processed samples are loaded, how the model is optimized and evaluated, and how training is structured to learn species-specific 3D structural patterns from LiDAR data.

- Implements a PointNet-style neural network for 3D point cloud segmentation
- Uses PyTorch
- Trains and evaluates the model on a tree point-cloud dataset
- Code inspired from : https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_cls.py - modifications include minimal adaptation for tree point-cloud segmentation

# 4. EDA

This section presents exploratory data analysis conducted to understand species distributions, structural variability, and correlations between biological and geometric and physical attributes. The insights gained guide preprocessing choices, model design decisions, and interpretation of learned patterns, with particular relevance for downstream 3D forest generation in game environments.

- Explores the dataset to understand the distribution, patterns, and potential preprocessing needs
- Provides insights into the dataset's characteristics to guide feature engineering and model development
- Explores potential correlations between physical structure (tree diameter and height) and biological attributes such as age
- Provides insights from several exploratory data analysis techniques with visualizations including: scatter plot, box pot, histograms, correlation matrices etc.
- Summarizes key findings and trends observed during exploratory data analysis as well discuss future impacts these can have in game design choices

# Run Instructions
Run the following in terminal.

For Window users:
* .\run_pipeline.ps1

For Linux users
* chmod +x run_pipeline.sh
* bash ./run_pipeline.sh

# References
[1] Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey
Qinfeng Zhua,b, Lei Fana, 1
, Ningxin Wenga, : https://arxiv.org/pdf/2308.12113

# Miscellaneous Notes
“The pip show command does not display license metadata for setuptools. The package is maintained by the Python Packaging Authority and licensed under MIT, as documented in its official repository and PyPI listing. All Python dependencies were audited for licensing. All required libraries use permissive licenses (MIT, BSD, Apache, PSF). Where license metadata was missing (e.g., setuptools), the license was manually verified from the official project repository.”
