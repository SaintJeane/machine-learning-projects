<div align="center">

# Image Classification System

**PyTorch computer vision, end to end** — from tensor fundamentals through transfer learning, Vision Transformers, experiment tracking, and live deployment on Hugging Face Spaces.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-Computer%20Vision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Deployment-F97316?style=flat-square&logo=gradio&logoColor=white)](https://www.gradio.app/)

</div>

---

## Live demos

| Demo | Description | Link |
|---|---|---|
| 🍕 **FoodVision Mini** | EfficientNetB2 classifying pizza, steak, or sushi | [Saint5-food-vision-mini.hf.space](https://saint5-food-vision-mini.hf.space) |
| 🍽️ **FoodVision Big** | EfficientNetB2 classifying all 101 classes of Food101 | [Saint5-foodvision-big.hf.space](https://saint5-foodvision-big.hf.space) |

Both are Gradio apps hosted on Hugging Face Spaces — no setup required, just open the link and upload an image.

---

## Overview

This directory is a progressive build-up of PyTorch computer vision skills, structured as a series of notebooks that each layer a new concept on top of the last:

- Clean PyTorch abstractions (`Dataset`, `DataLoader`, training loops)
- Reproducible experimentation with fixed seeds
- Transfer learning with pretrained CNN and Transformer backbones
- Experiment tracking and model comparison via TensorBoard
- Performance-aware deployment decisions (accuracy vs. speed vs. model size)

The project culminates in two deployed models — **FoodVision Mini** (3 food classes) and **FoodVision Big** (101 food classes) — each shipped as a Gradio app on Hugging Face Spaces.

---

## Repository structure

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

> Note: notebook 05 ("Modular Helper Functions") isn't included as a standalone notebook in this folder — its output is the reusable helper scripts (data setup, training/testing engines, seed utilities) consumed by notebooks 06 onward.

---

## Notebooks

| # | Notebook | Covers |
|---|---|---|
| 01 | [PyTorch Fundamentals](01_PyTorch_tutorial.ipynb) | Core tensor operations and autograd mechanics |
| 02 | [Classification Pipeline](02_Classification_Using_PyTorch.ipynb) | End-to-end supervised workflow: datasets, loaders, training loop, evaluation |
| 03 | [Computer Vision Basics](03_PyTorch_Computer_Vision.ipynb) | `torchvision` datasets, transforms, and a from-scratch CNN (TinyVGG) |
| 04 | [Custom Dataset Patterns](04_PyTorch_Computer_Vision_Customized_Dataset.ipynb) | Custom `Dataset`/`DataLoader` classes for non-standard data layouts |
| 06 | [Transfer Learning](06_PyTorch_transfer_learning.ipynb) | Fine-tuning pretrained backbones; layer freezing/unfreezing strategies |
| 07 | [Experiment Tracking](07_Experiment_tracking.ipynb) | Logging and comparing runs with TensorBoard |
| 08 | [Vision Transformers (ViT)](08_ViT_for_foodvision_using_PyTorch.ipynb) | Applying ViT-B/16 to food image classification |
| 09 | [Model Deployment](09_model_deployment.ipynb) | EffNetB2 vs. ViT tradeoff analysis, Gradio demo, Hugging Face Spaces deployment |

---

## Results: EfficientNetB2 vs. ViT-B/16

Both models were fine-tuned as feature extractors on a 20% subset of Food101 (pizza, steak, sushi) and benchmarked on CPU inference, per notebook 09:

| Metric | EfficientNetB2 | ViT-B/16 |
|---|---|---|
| Test accuracy | 86.88% | **98.47%** |
| Test loss | 0.2811 | **0.0644** |
| Parameters | **7.7M** | 85.8M |
| Model size | **29 MB** | 327 MB |
| Avg. CPU inference time | **0.106 s/pred** | 0.603 s/pred |

**Takeaway:** ViT is meaningfully more accurate, but at ~11x the parameters, ~11x the model size, and ~5.7x the inference latency of EffNetB2. Against the project's original deployment targets (≥95% accuracy, ≥30 FPS / <0.03s per prediction), neither model hits the latency bar on CPU — a good illustration of why the accuracy/speed tradeoff, not raw accuracy, is what actually drove the deployment choice. **EfficientNetB2 was shipped** for both live demos above, since it stays usably fast on CPU-only Hugging Face Spaces hardware.

---

## Environment & tools

**Frameworks:** PyTorch, TorchVision, TensorBoard, scikit-learn, OpenCV, Gradio
**Data/plotting:** NumPy, Pandas, Matplotlib
**Device handling:** dynamic `cuda`/`cpu` selection; notebooks run GPU-accelerated when available (developed on Google Colab's free-tier T4)

---

## Datasets

- **[Food101](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)** (`torchvision.datasets`) — used for transfer learning and both deployment demos, including a hand-curated 3-class (pizza/steak/sushi) subset for FoodVision Mini
- **[FashionMNIST](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.FashionMNIST.html)** — used for the introductory CNN/TinyVGG notebooks

Datasets are downloaded and prepared programmatically within the notebooks.

---

## Models

- **TinyVGG** — small CNN built from scratch, educational baseline (notebook 03)
- **EfficientNetB0 / EfficientNetB2** — pretrained feature extractors, fine-tuned classifier head
- **ViT-B/16** — pretrained Vision Transformer, fine-tuned classifier head, compared directly against EfficientNetB2 for deployment

---

## Deployment

Notebook 09 packages the winning model (EfficientNetB2) into a Gradio app and ships it to Hugging Face Spaces, in two versions:

- **FoodVision Mini** — 3-class classifier (pizza/steak/sushi), demo app structured as `app.py` + `model.py` + saved `.pth` weights + `requirements.txt`
- **FoodVision Big** — full 101-class Food101 classifier, same structure plus a `class_names.txt`

The demo source lives inside notebook 09 (cells that write `app.py`/`model.py` via `%%writefile`) rather than as separate committed files, since the packaged app folders are zipped and pushed directly to their respective Hugging Face Spaces.

---

## Credit

Core training/evaluation helper functions (data setup, train/test engines, reproducibility seeding) are adapted from [`mrdbourke/pytorch-deep-learning`](https://github.com/mrdbourke/pytorch-deep-learning), following the accompanying deployment walkthrough.

---

## License

Apache 2.0 — see [LICENSE](../LICENSE) at the repository root.