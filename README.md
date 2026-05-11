# Machine Learning Projects

<p align="center">

  <!-- Core Frameworks -->
  <img src="https://img.shields.io/badge/PyTorch-FF4C4C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/TorchVision-FF4C4C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-4B8BBE?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F28500?style=flat&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=black"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Gradio-FF6F00?style=flat&logo=gradio&logoColor=white"/>

  <!-- Models -->
  <img src="https://img.shields.io/badge/Vision%20Transformer-9C27B0?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/EfficientNet-607D8B?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/DistilBERT-FFC107?style=flat&logo=huggingface&logoColor=black"/>

  <!-- Concepts & Meta -->
  <img src="https://img.shields.io/badge/Computer%20Vision-2196F3?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-26A69A?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transfer%20Learning-00BCD4?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-E91E63?style=flat&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-9E9E9E?style=flat&logoColor=white"/>

</p>


This repository showcases a collection of end-to-end machine learning projects demonstrating practical experience across experiment tracking and model development, training, evaluation, and deployment.

The work emphasizes reproducibility, clean experimentation, and best practices, with reusable templates and utility scripts that support real-world machine learning workflows.

The projects span multiple domains, including:
- **Computer Vision** — image classification pipelines (e.g., food image classification, Neural Networks, Vision Transformers)
- **Natural Language Processing (NLP)** — binary text classification and dataset creation

## Technical Focus

- Transfer learning and model fine-tuning
- Custom dataset creation and preprocessing
- Training and evaluation workflows in PyTorch.
- Experiment tracking using `TensorBoard`.
- Experimentation with various computer vision model architectures (ViT, EffNetB2).
- Model deployment and inference pipelines using Gradio
- Hugging Face–based NLP text classification workflows

## Repository Structure

```text
├── image_classification_system_project/
│   ├── 01_PyTorch_tutorial.ipynb
│   ├── 02_Classification_Using_PyTorch.ipynb
│   ├── 03_PyTorch_Computer_Vision.ipynb
│   ├── 04_PyTorch_Computer_Vision_Customized_Dataset.ipynb
│   ├── 06_PyTorch_transfer_learning.ipynb
|   ├── 07_Experiment_tracking
│   ├── 08_ViT_for_foodvision_using_PyTorch.ipynb
│   ├── 09_model_deployment.ipynb
│   └── README.md
├── binary_text_classifier_system/
│   ├── huggingface_text_classification.ipynb
│   ├── huggingface_food_not_food_image_caption_dataset_creation.ipynb
│   └── README.md
├── README.md
├── .gitignore
└── LICENSE
```