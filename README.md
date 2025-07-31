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

## data prep

### ups env:
currently using local builds of wire-cell and larereco
```bash
# in the sl7 container:
source /exp/sbnd/app/users/yuhw/wire-cell-toolkit/setup.sh
```

### python env:
```bash
# gpvm
source /exp/sbnd/app/users/yuhw/dl-clustering/venv/bin/activate
# eaf
source /exp/sbnd/app/users/yuhw/dl-clustering/venv_eaf/bin/activate
```

### prep
setup the two batch scripts then run commands below:
```bash
# ups env
./batch_run_fcl.sh
# python env
./batch_run_labeling.sh
```

### train
```bash
./train.sh
```

### val
```bash
./val.sh
```


### Data Format:

rec:
points: [x, y, z, q, blob_idx]
ppedges: [head, tail, dist]
blobs: [q, ncorners, x_i, y_i, z_i, ..., x_ncorners, ...]
