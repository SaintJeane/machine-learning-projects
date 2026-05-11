# Food vs. Not Food Text Classifier System

This subfolder contains a binary text classifier system leveraging the Huggingface ecosystem for dataset creation and model training on text data.

## Contents

- **[huggingface_food_not_food_image_caption_dataset_creation.ipynb](https://github.com/SaintJeane/machine-learning-projects/blob/main/huggingface_project/huggingface_food_not_food_image_caption_dataset_creation.ipynb)**  
  Jupyter notebook for generating an image captioning dataset that distinguishes food from non-food items. Includes code for scraping, preprocessing, and dataset formatting compatible with Huggingface Datasets.

- **[huggingface_text_classification.ipynb](https://github.com/SaintJeane/machine-learning-projects/blob/main/huggingface_project/huggingface_text_classification.ipynb)**  
  Jupyter notebook for building and training a text classification model using Huggingface Transformers. Covers data loading, preprocessing, model selection, fine-tuning, evaluation, and deployment.


## Notebooks Overview

### 1. Image Caption Dataset Creation
- Synthetic generation of text dataset for food and non-food texts.
- Assigns captions for food/not-food detection.
- Prepares and exports the dataset in Huggingface-compatible format.

### 2. Text Classification
- Loads text data for classification tasks.
- Preprocesses and tokenizes text using Huggingface `distilbert`.
- Selects and fine-tunes HuggingFace Transformer model `distilbert-base-uncased`.
- Evaluates performance and saves the results and the trained model.
- Deploy the trained model to [HuggingFace Spaces using Gradio interface](https://huggingface.co/spaces/Saint5/hg_tutorial_food_not_food_text_classifier_demo).

## Results

Results, sample outputs, and evaluation metrics can be found in the corresponding notebook outputs.  

## References

- [Huggingface Transformers Documentation](https://huggingface.co/docs/transformers)
- [Huggingface Datasets Documentation](https://huggingface.co/docs/datasets)