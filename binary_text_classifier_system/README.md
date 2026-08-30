<div align="center">

# Food / Not-Food Text Classifier

**Synthetic dataset creation → DistilBERT fine-tuning → live Gradio deployment on Hugging Face Spaces**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![Datasets](https://img.shields.io/badge/🤗%20Datasets-Hub-FFD21E?style=flat-square)](https://huggingface.co/docs/datasets)
[![Gradio](https://img.shields.io/badge/Gradio-Deployment-F97316?style=flat-square&logo=gradio&logoColor=white)](https://www.gradio.app/)

</div>

---

## Live demo

**[Food / Not-Food Text Classifier — Hugging Face Space](https://huggingface.co/spaces/Saint5/hg_tutorial_food_not_food_text_classifier_demo)**

Type any short piece of text and the model predicts whether it describes food or not, with a confidence score.

---

## Overview

A binary text classification system built end-to-end on the Hugging Face ecosystem: a synthetic dataset is generated, uploaded to the Hub, used to fine-tune DistilBERT, then the trained model is wrapped in a Gradio app and deployed programmatically to Hugging Face Spaces.

## Contents

| Notebook | Purpose |
|---|---|
| [`huggingface_food_not_food_image_caption_dataset_creation.ipynb`](huggingface_food_not_food_image_caption_dataset_creation.ipynb) | Builds and uploads the food/not-food caption dataset |
| [`huggingface_text_classification.ipynb`](huggingface_text_classification.ipynb) | Fine-tunes DistilBERT on the dataset, evaluates it, and deploys it as a Gradio Space |

---

## 1. Dataset creation

- 250 short image-caption-style sentences (125 `food`, 125 `not_food`), generated with [Mistral Chat](https://chat.mistral.ai/chat) as synthetic training data — a fast way to prototype a classifier before scaling to real/scraped data.
- Captions are assembled into a Pandas `DataFrame`, shuffled, and converted to a Hugging Face `Dataset` with `Dataset.from_pandas()`.
- Pushed to the Hub as a public dataset for reuse.

> **Note:** the notebook uploads to and loads from the `mrdbourke/learn_hf_food_not_food_image_captions` namespace (the tutorial author's dataset repo) rather than a personal one — worth re-pointing at your own Hugging Face username if you want the dataset hosted under your own account.

## 2. Fine-tuning DistilBERT

- **Base model:** `distilbert/distilbert-base-uncased` (≈67M parameters), fine-tuned via `AutoModelForSequenceClassification` with a 2-label classification head (`food` / `not_food`)
- **Split:** 200 train / 50 test (80/20, seeded)
- **Hyperparameters:** batch size 32, learning rate 1e-4, 10 epochs, best-model checkpointing on eval loss
- **Training time:** ~96 seconds end-to-end (GPU)

### Results

| Metric | Value |
|---|---|
| Final train loss | 0.0495 |
| Test loss | 0.0005 |
| Test accuracy | **100%** (50/50) |

> The dataset is small and synthetically generated with clearly distinct vocabulary between classes, which is why accuracy converges to 100% — a good demonstration of the fine-tuning workflow, but not a benchmark that would hold on messier, real-world text. Worth keeping in mind if extending this to a harder dataset.

- Trained model pushed to the Hub: [`Saint5/hg_tutorial_food_not_food_text_classifier_distilbert_base_uncased`](https://huggingface.co/Saint5/hg_tutorial_food_not_food_text_classifier_distilbert_base_uncased)

## 3. Deployment

The trained model is wrapped in a Gradio `app.py`, packaged with a `requirements.txt`, and uploaded programmatically via the `huggingface_hub` Python API (`create_repo` + `upload_folder`) to a Hugging Face Space — no manual file uploads through the browser.

---

## Environment & tools

**Libraries:** `transformers`, `datasets`, `evaluate`, `accelerate`, `gradio`, `torch`, `scikit-learn`, `pandas`
**Training hardware:** Google Colab (GPU runtime)

---

## References

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets)