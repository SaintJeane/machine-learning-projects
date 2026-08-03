# Image Classification Project

Practical, hands-on computer vision experiments in PyTorch: from PyTorch fundamentals to transfer learning, Vision Transformers (ViT), experiment tracking, and a minimal deployment demo.

This directory contains Jupyter notebooks that walk through dataset preparation, training loops, model evaluation, experiment tracking (TensorBoard), and a simple Gradio-based inference demo.

The emphasis is on:
- Clean PyTorch abstractions (Dataset, DataLoader, training loops)
- Reproducible experimentation
- Practical use of pretrained models
- Experiment tracking of various models' train and test accuracies and losses.
- Performance-aware deployment considerations

---

## File structure (this directory)
```
image_classification_system_project/
├─ 01_PyTorch_tutorial.ipynb
├─ 02_Classification_Using_PyTorch.ipynb
├─ 03_PyTorch_Computer_Vision.ipynb
├─ 04_PyTorch_Computer_Vision_Customized_Dataset.ipynb
├─ 06_PyTorch_transfer_learning.ipynb
├─ 07_Experiment_tracking.ipynb
├─ 08_ViT_for_foodvision_using_PyTorch.ipynb
├─ 09_model_deployment.ipynb
└─ README.md
```

---

## Notebooks Overview

* **[01 - PyTorch Fundamentals](01_pytorch_tutorial.ipynb)**: Entails core PyTorch primitives, tensor operations, and autograd mechanics. It's all about PyTorch's basics.

* **[02 - Classification Pipeline](02_classification_using_pytorch.ipynb)**: Entails end-to-end supervised learning workflow: datasets, data loaders, training loop, and evaluation structure.

* **[03 - Computer Vision Basics](03_pytorch_computer_vision.ipynb)**: Vision datasets, transforms, visualization, and CNN-based experimentation using `torchvision`.

* **[04 - Custom Dataset Patterns](04_pytorch_computer_vision_customized_dataset.ipynb)**: Implementing custom dataset and DataLoader classes for non-standard data layouts.

* **[05 - Creating Modular Helper Functions](___)**: Creating modular scripts for re-using in other projects (logic is applied in the following projects).

* **[06 - Transfer Learning](06_pytorch_transfer_learning.ipynb)**: Fine-tuning pretrained backbones and exploring freezing/unfreezing strategies for efficient training.

* **[07 - Experiment Tracking](07_experiment_tracking.ipynb)**: Experiment tracking and comparing the results logs of the two image classifier models using `TensorBoard`.

* **[08 - Vision Transformers (ViT)](08_ViT_for_foodvision_using_pytorch.ipynb)**: Applying transformer-based architectures to food image classification tasks.

* **[09 - Model Deployment](09_model_deployment.ipynb)**: Deployment-oriented experiments, accuracy vs latency comparisons, and a Gradio inference demo.

---

## Environment & Runtime

* **Frameworks and Libraries**

  * PyTorch
  * TorchVision
  * TensorBoard
  * OpenCV - Python
  * Scikit-Learn
  * Gradio
  * Numpy
  * Pandas
  * Matplotlib

* **Device handling**

  * Dynamic device selection (`cuda` if available, otherwise `cpu`)
  * Experiments are GPU-accelerated when CUDA is available (recommendable)

---

## Datasets Used

* **[torchvision food 101 dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)**

  * Torchvision 101 food classes Food Dataset
  * Used in transfer learning and deployment experiments
  * Train/test directory-based structure

* **[FashionMNIST Dataset](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.FashionMNIST.html#fashionmnist)**

  * Standard torchvision dataset
  * Used for introductory computer vision experiments

Dataset downloads and preparation are handled programmatically within notebooks or via helper utilities.

---

## Models & Architectures

* **CNN-based models** (introductory CV notebooks)
    - **TinyVGG** - creating from scratch for image classification - educational baseline.
    - **EfficientNetB0** - using pretrained model weights and architecture for inference.
  
* **EfficientNet-B2**
  * Used as a pretrained feature extractor
  * Evaluated for accuracy–latency tradeoffs
* **Vision Transformer (ViT-B/16)**
  * Applied to food image classification
  * Compared directly against EfficientNet during deployment experiments

---

## Deployment-Oriented Experiment

The [final notebook](09_model_deployment.ipynb) focuses explicitly on deployment constraints:

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