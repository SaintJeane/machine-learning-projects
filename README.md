# Machine Learning Projects

<p align="center">

  <!-- Language -->
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  </a>

  <!-- Frameworks -->
  <a href="https://pytorch.org/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://pytorch.org/vision/stable/" target="_blank">
    <img src="https://img.shields.io/badge/TorchVision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="TorchVision"/>
  </a>
  <a href="https://www.tensorflow.org/tensorboard" target="_blank">
    <img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorBoard"/>
  </a>
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  </a>

  <!-- Development Tools -->
  <a href="https://jupyter.org/" target="_blank">
    <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"/>
  </a>
  <a href="https://colab.research.google.com/" target="_blank">
    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black" alt="Google Colab"/>
  </a>
  <a href="https://huggingface.co/" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  </a>
  <a href="https://www.gradio.app/" target="_blank">
    <img src="https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  </a>

  <!-- Models -->
  <a href="https://arxiv.org/abs/2010.11929" target="_blank">
    <img src="https://img.shields.io/badge/Vision%20Transformer-8E44AD?style=flat-square&logoColor=white" alt="Vision Transformer"/>
  </a>
  <a href="https://arxiv.org/abs/1905.11946" target="_blank">
    <img src="https://img.shields.io/badge/EfficientNet-607D8B?style=flat-square&logoColor=white" alt="EfficientNet"/>
  </a>
  <a href="https://huggingface.co/docs/transformers/model_doc/distilbert" target="_blank">
    <img src="https://img.shields.io/badge/DistilBERT-FFC107?style=flat-square&logo=huggingface&logoColor=black" alt="DistilBERT"/>
  </a>

  <!-- Domains & Concepts -->
  <a href="https://en.wikipedia.org/wiki/Computer_vision" target="_blank">
    <img src="https://img.shields.io/badge/Computer%20Vision-2196F3?style=flat-square&logoColor=white" alt="Computer Vision"/>
  </a>
  <a href="https://en.wikipedia.org/wiki/Natural_language_processing" target="_blank">
    <img src="https://img.shields.io/badge/NLP-26A69A?style=flat-square&logoColor=white" alt="Natural Language Processing"/>
  </a>
  <a href="https://en.wikipedia.org/wiki/Transfer_learning" target="_blank">
    <img src="https://img.shields.io/badge/Transfer%20Learning-00BCD4?style=flat-square&logoColor=white" alt="Transfer Learning"/>
  </a>
  <a href="https://en.wikipedia.org/wiki/Deep_learning" target="_blank">
    <img src="https://img.shields.io/badge/Deep%20Learning-E91E63?style=flat-square&logoColor=white" alt="Deep Learning"/>
  </a>

  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-9E9E9E?style=flat-square&logoColor=white" alt="Apache License 2.0"/>
  </a>

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
│   ├── 01_pytorch_tutorial.ipynb
│   ├── 02_classification_using_pytorch.ipynb
│   ├── 03_pytorch_computer_vision.ipynb
│   ├── 04_pytorch_computer_vision_customized_dataset.ipynb
│   ├── 06_pytorch_transfer_learning.ipynb
|   ├── 07_experiment_tracking
│   ├── 08_ViT_for_foodvision_using_pytorch.ipynb
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