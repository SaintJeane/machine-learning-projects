# PyTorch Deep Learning

This directory contains a structured, hands-on PyTorch portfolio demonstrating end-to-end deep learning workflows.  
The notebooks progress from PyTorch fundamentals to applied computer vision, transfer learning, Vision Transformers (ViT), and a deployment-focused inference demo.

The emphasis is on:
- Clean PyTorch abstractions (Dataset, DataLoader, training loops)
- Reproducible experimentation
- Practical use of pretrained models
- Performance-aware deployment considerations

---

## Notebook Overview

```text
01_PyTorch_tutorial.ipynb
02_Classification_Using_PyTorch.ipynb
03_PyTorch_Computer_Vision.ipynb
04_PyTorch_Computer_Vision_Customized_Dataset.ipynb
06_PyTorch_transfer_learning.ipynb
08_ViT_for_foodvision_using_PyTorch.ipynb
09_model_deployment.ipynb
```

### High-level progression

* **01 — PyTorch Fundamentals**
  Core PyTorch primitives, tensor operations, and autograd mechanics.

* **02 — Classification Pipeline**
  End-to-end supervised learning workflow: datasets, data loaders, training loop, and evaluation structure.

* **03 — Computer Vision Basics**
  Vision datasets, transforms, visualization, and CNN-based experimentation using `torchvision`.

* **04 — Custom Dataset Patterns**
  Implementing custom `Dataset` and `DataLoader` classes for non-standard data layouts.

* **06 — Transfer Learning**
  Fine-tuning pretrained backbones and exploring freezing/unfreezing strategies for efficient training.

* **08 — Vision Transformers (ViT)**
  Applying transformer-based architectures to food image classification tasks.

* **09 — Model Deployment**
  Comparing EfficientNet and ViT models under accuracy and latency constraints, and demonstrating a minimal inference/deployment workflow.

---

## Environment & Runtime

* **Frameworks**

  * PyTorch: `2.6.0+cu124`
  * TorchVision: `0.21.0+cu124`

* **Device handling**

  * Dynamic device selection (`cuda` if available, otherwise `cpu`)
  * Experiments are GPU-accelerated when CUDA is available

---

## Datasets Used

* **pizza_steak_sushi_20_percent**

  * 20% of Torchvision 101 Food Dataset
  * Used in transfer learning and deployment experiments
  * Train/test directory-based structure

* **FashionMNIST**

  * Standard torchvision dataset
  * Used for introductory computer vision experiments

Dataset downloads and preparation are handled programmatically within notebooks or via helper utilities.

---

## Models & Architectures

* **CNN-based models** (introductory CV notebooks)
* **EfficientNet-B2**

  * Used as a pretrained feature extractor
  * Evaluated for accuracy–latency tradeoffs
* **Vision Transformer (ViT-B/16)**

  * Applied to food image classification
  * Compared directly against EfficientNet during deployment experiments

---

## Deployment-Oriented Experiment

The final notebook focuses explicitly on deployment constraints:

* **Target accuracy:** ≥ 95%
* **Target inference speed:** ≥ 30 FPS
  (≈ < 0.03s per prediction)

Models are evaluated not only on accuracy but also on suitability for real-time inference scenarios.

---

## External Utilities & Reuse

Several notebooks reuse modular helper code sourced from
[`mrdbourke/pytorch-deep-learning`](https://github.com/SaintJeane/pytorch-deep-learning), including:

* Data setup utilities
* Training and evaluation engines
* Reproducibility helpers (seed setting)
* Visualization utilities

This mirrors real-world workflows where shared training infrastructure is abstracted into reusable modules.

---