# Model Training and Parameter Search

This folder contains two complementary Python scripts that performs parameter search and fine-tune a **TinyBERT** model.

- `paraSearch.py` – performs **hyper-parameter search** with Ray Tune, Wandb, and Huggingface 🤗 Transformers’ `Trainer` interface.
- `trainer.py` – **fine-tunes** the model with the best parameters found (Already using the best parameter we found through parasearch) and produces detailed evaluation reports.
- `requirements.txt` – exact Python package versions used during development (Python 3.11.9).

## Notes

- The requirements.txt does not include the Pytorch installation since you need to choose the best version that fit your device. To install Pytorch, please follw the official installation [guide](https://pytorch.org/get-started/locally/).

- This following package version includes the version we proved to work with the current code. However, the dependabot has complained about the security issue and automatically updated the package to the higher safe version in the requirements.txt. If you encounter any issues, please try to downgrade the package version to the one we used in the code. The version we used is:

  ```bash
  numpy==1.26.4
  datasets==3.0.0
  transformers==4.44.2
  scikit-learn==1.5.2
  wandb==0.18.0
  ray==2.35.0
  tokenizers==0.19.1
  pyarrow==17.0.0
  accelerate>=0.21.0
  ```
