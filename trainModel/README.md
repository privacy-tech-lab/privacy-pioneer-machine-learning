# Model Training and Parameter Search

This folder contains two complementary Python scripts that performs parameter search and fine-tune a **TinyBERT** model.

- `paraSearch.py` – performs **hyper-parameter search** with Ray Tune, Wandb, and Huggingface 🤗 Transformers’ `Trainer` interface.
- `trainer.py` – **fine-tunes** the model with the best parameters found (Already using the best parameter we found through parasearch) and produces detailed evaluation reports.
- `requirements.txt` – exact Python package versions used during development (Python 3.11.9).

## Notes

The requirements.txt does not include the Pytorch installation since you need to choose the best version that fit your device. To install Pytorch, please follw the official installation [guide](https://pytorch.org/get-started/locally/).
