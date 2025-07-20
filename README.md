**EEG Classification with NeuroGATE**

This repository contains code to run and evaluate the NeuroGATE deep learning model on EEG data. It provides all necessary data loaders, preprocessing pipelines, model definition, training utilities, and an example notebook demonstrating a complete experiment.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)

---

## Features

* **EEG Preprocessing Pipelines**: Configurable filtering, resampling, and artifact rejection.
* **PyTorch Dataset**: Ready-to-use `torch.utils.data.Dataset` for EEG file formats.
* **NeuroGATE Model**: Reference implementation of the NeuroGATE architecture for classification tasks.
* **Training Utilities**: High-level training loop.

---

## Installation

This package is intended for local development and testing. To install:

```bash
# Clone the repository
git clone https://github.com/dll-ncai/NeuroGate_EEG_Classification.git
cd NeuroGate_EEG_Classification

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\\Scripts\\activate  # Windows

# Install in editable mode
pip install -e .
```

---

## Quick Start

All code usage is demonstrated in the provided Jupyter notebook. After installation, launch:

```bash
jupyter notebook example.ipynb
```

The notebook walks through:

1. Loading and preprocessing EEG data
2. Initializing and training the NeuroGATE model
3. Inspecting results and metrics

---

## Project Structure

```plaintext
NeuroGate_EEG_Classification/
|-- neurogate_eeg
|   |-- datasets
|   |   |-- channels.py
|   |   |-- dataset.py
|   |   |-- defaults.py
|   |   |-- getfiles.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- pytordataset.py
|   |   `-- sc_pipeline.py
|   |-- models
|   |   |-- __init__.py
|   |   `-- neurogate.py
|   |-- train
|   |   |-- callbacks.py
|   |   |-- __init__.py
|   |   |-- misc.py
|   |   |-- retrieve.py
|   |   |-- store.py
|   |   |-- train.py
|   |   `-- xloop.py
|   `-- __init__.py
|-- example.ipynb
|-- README.md
`-- setup.py
```

---

## Dependencies

* Python3
* numpy
* pandas
* mne
* torch
* scikit-learn
* tqdm
* matplotlib

