# DL-CLUS: Deep Learning Clustering based on Wire-Cell 3D output

This package provides deep learning tools for clustering 3D points in the SBND (Short-Baseline Near Detector) experiment.

## Overview

DL-CLUS performs neutrino interaction clustering in LArTPC data using deep learning techniques. The package includes:

- Data preprocessing tools
- Truth labeling utilities
- Deep learning models for clustering

## Installation

```bash
# Clone the repository
git clone https://github.com/username/dl-clus.git
cd dl-clus

# Install the package
pip install -e .
```

## Usage

### Data Preparation

Use the labeling utilities to prepare your data:

```bash
python dl-clus/prep/labeling.py --tru-prefix /path/to/truth/files \
                               --rec-prefix /path/to/reco/files \
                               --out-prefix /path/to/output/files \
                               --entries 0-10
```

### FCL Files

The `fcl` directory contains FHiCL configuration files for running WireCell with different settings:

- `wcls-img-clus.fcl`: Configuration for WireCell imaging and clustering
- `celltree_sbnd_apa0.fcl` and `celltree_sbnd_apa1.fcl`: Configurations for the two APAs

## License

This project is licensed under the terms of the MIT license.
