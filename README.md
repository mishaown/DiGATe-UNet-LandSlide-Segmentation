# DiGATe-UNet: Lightweight Dual-Stream Framework for Landslide Segmentation

This repository contains the implementation of our lightweight dual-stream Siamese framework for landslide segmentation from remote sensing imagery. The model integrates optical and topographical data fusion, an adaptive decoder with lightweight cross-attention, gated fusion, and deep supervision to achieve accurate boundary delineation and robust performance across diverse landscapes.

* 📜 Our manuscript has been accepted by the CMC-Computers, Materials & Continua journal. 

* 📌 **Current content:** Full training pipeline, model implementation, and benchmark datasets integration.
---

## Requirements

> Tested on **Python 3.10** with CUDA 12.x.

Minimal runtime dependencies (the ones actually used by the code/notebook):

```txt
numpy==1.26.4
torch==2.6.0
torchvision==0.21.0
timm==1.0.19
segmentation-models-pytorch==0.4.0
opencv-python==4.11.0.86
Pillow==11.1.0
h5py==3.13.0
tqdm==4.67.1
matplotlib==3.10.1
scikit-image==0.25.2
```

**Optional (only if you use certain SMP backbones that rely on them):**

```txt
efficientnet-pytorch==0.7.1
pretrainedmodels==0.7.4
```

---

## Installation

1. **Create env (recommended)**

```bash
conda create -n digate python=3.10 -y
conda activate digate
```

2. **Install PyTorch first** (match your CUDA):

```bash
# Example for CUDA 12.x wheels
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
```

> If you’re on CPU or a different CUDA, use the command from the official PyTorch site for your setup.

3. **Install the rest**

```bash
pip install -r requirements.txt
```

---

## Quick Start (Inference) Weights required*

1. Place the Bijie dataset (or your data) under the expected paths used in `inference.ipynb`.
2. Open the notebook:

```bash
jupyter notebook inference.ipynb
```

3. Run all cells to produce masks and basic metrics/visualizations.

---

## Dataset
* Bijie landslide dataset (examples used in the notebook).
* Additional datasets and loaders will be wired in with the training pipeline update.

---
## Project Structure (current)

```
└── \datset\                                  # Dataset preprocessing
└── \models\                                  # Models, Loss and metrics
└── \rrain\                                   # Training
└── \utils\                                   # Evaluate and plot
└── LS_segmentation_landslide4sense.ipynb     # Train on LandSlide4Sense dataset
├── inference.ipynb                           # Example inference workflow on Bijie
├── backbones.ipynb                           # Test computational efficieny on trained weigths
└── LS_segmentation_bijie.ipynb               # Train on Bijie dataset
```

---
<!--  
## Citation

If you find this useful, please cite (placeholder):

```bibtex
@misc{digate_unet_2025,
  title  = {DiGATe-UNet: Lightweight Dual-Stream Framework for Landslide Segmentation},
  author = {Your Name},
  year   = {2025},
  note   = {GitHub repository}
}
```

---

## License

MIT (or your chosen license).
-->
