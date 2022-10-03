# Privacy Pioneer Machine Learning

## 1. What Is This?

Everything machine learning-related to [Privacy Pioneer](https://github.com/privacy-tech-lab/privacy-pioneer).

## 2. What libraries, technologies, and techniques are we using?

- Training/validation/test set data
    - The datasets we are using are located within the [Privacy Pioneer Google Drive Folder](https://drive.google.com/drive/folders/1GyJDTYrsEcRZ-tD-pAScjE7yOL9Z9O-j?usp=sharing), and are also in this repo under the `annotatedData` folder. We are using a 80% training, 10% validation, 10% test split of the data. The test dataset is set up to be the data that was labeled by multiple independant labelers. See also [Hugging Face](https://huggingface.co/dgoldelman) for the datasets used in the generation of our models.
- Hugging Face
    - [Hugging Face](https://huggingface.co/) is a machine learning library, ML/AI community, and dedicated API that that is set up to assist with the creation, storage, and distribution of machine learning programs and datasets.
- Google Colab
    - [Google Colab](https://colab.research.google.com/) is a experimental framework for Jupiter Notebooks to be run on the cloud on GPUs and TPUs that are able to run independantly of your own computer's runtimes. This enables us to quickly build and test ML models using minimal resources.
- Base Model Details
    - We primarilly make use of [TinyBert](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) and [bert-base-uncased](https://huggingface.co/bert-base-uncased) as the base models which we retrain for our specific use case. We use TinyBert as the main model because it is only 59mb, which we deem as an acceptable size for achieving accuracy while remaining small enough to be loaded into a browser extension. We also explored [Knowledge Distillation](https://analyticsindiamag.com/a-beginners-guide-to-knowledge-distillation-in-deep-learning/) from bert-base-uncased to TinyBert to better achieve accuracy at a smaller model size.
- How do the different scripts, frameworks, data interact?
    - Our datasets are located on Hugging Face, currently under [Daniel's account](https://huggingface.co/dgoldelman) (this will be changed). Our files, scripts, and models are within the Privacy Pioneer Google Drive Folder's [Machine Learning](https://drive.google.com/drive/folders/1tjah6qy8JKf3RmI-ZxnKceiXuGahysE5?usp=sharing) section. Our output models are also under Daniel's account.
- Where is the model you use in Privacy Pioneer?
    - The folder [./convertMultiModel/multitaskModelForJSWeb](./convertMultiModel/multitaskModelForJSWeb) is our model that we load into the Privacy Pioneer browser extension.

## 3. Thank You!

<p align="center"><strong>We would like to thank our financial supporters!</strong></p><br>

<p align="center">Major financial support provided by Google.</p>

<p align="center">
  <a href="https://research.google/outreach/research-scholar-program/recipients/">
    <img class="img-fluid" src="./google_logo.png" height="80px" alt="Google Logo">
  </a>
</p>

<p align="center">Additional financial support provided by the Anil Fernando Endowment and Wesleyan University.</p>

<p align="center">
  <a href="https://www.wesleyan.edu/mathcs/cs/index.html">
    <img class="img-fluid" src="./wesleyan_shield.png" height="70px" alt="Wesleyan University Logo">
  </a>
</p>

<p align="center">Conclusions reached or positions taken are our own and not necessarily those of our financial supporters, its trustees, officers, or staff.</p>

##

<p align="center">
  <a href="https://privacytechlab.org/"><img src="./plt_logo.png" width="200px" height="200px" alt="privacy-tech-lab logo"></a>
<p>
