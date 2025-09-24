# Huggingface Project

This subfolder contains projects leveraging the Huggingface ecosystem for dataset creation and model training on image and text data.

## Contents

- **huggingface_food_not_food_image_caption_dataset_creation.ipynb**  
  Jupyter notebook for generating an image captioning dataset that distinguishes food from non-food items. Includes code for scraping, preprocessing, and dataset formatting compatible with Huggingface Datasets.

- **huggingface_text_classification.ipynb**  
  Jupyter notebook for building and training a text classification model using Huggingface Transformers. Covers data loading, preprocessing, model selection, fine-tuning, evaluation, and export.

## Getting Started

### Prerequisites

- Python 3.8+
- `transformers`
- `datasets`
- `torch`
- `jupyter`
- (Optional) `scikit-learn`, `matplotlib`, and other supporting libraries

Install dependencies:
```bash
pip install transformers datasets torch jupyter scikit-learn matplotlib
```

### Usage

Clone the repository and launch Jupyter:
```bash
git clone https://github.com/SaintJeane/machine-learning-projects.git
cd machine-learning-projects/huggingface_project
jupyter notebook
```
Open the desired notebook and follow the instructions inside.

## Notebooks Overview

### 1. Image Caption Dataset Creation
- Scrapes or loads image data.
- Assigns captions for food/not-food detection.
- Prepares and exports the dataset in Huggingface-compatible format.

### 2. Text Classification
- Loads text data for classification tasks.
- Preprocesses and tokenizes text using Huggingface tools.
- Selects and fine-tunes a Transformer model.
- Evaluates performance and saves results.

## Results

Results, sample outputs, and evaluation metrics can be found in the corresponding notebook outputs.  
To contribute results or enhancements, please fork the repository and open a pull request.

## Contributing

Pull requests and suggestions are welcome! Please ensure any new notebooks:
- Use Huggingface datasets or models.
- Include clear instructions and comments.
- Save outputs and results where appropriate.

## License

This project is licensed under the MIT License.

## References

- [Huggingface Transformers Documentation](https://huggingface.co/docs/transformers)
- [Huggingface Datasets Documentation](https://huggingface.co/docs/datasets)