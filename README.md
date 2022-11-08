# Privacy Pioneer Machine Learning

## 1. What Is This?

Everything machine learning-related to [Privacy Pioneer](https://github.com/privacy-tech-lab/privacy-pioneer).

## 2. What libraries, technologies, and techniques are we using?

- Training/validation/test set data
    - The datasets we are using are located within the [Privacy Pioneer Google Drive Folder](https://drive.google.com/drive/folders/1GyJDTYrsEcRZ-tD-pAScjE7yOL9Z9O-j?usp=sharing), and are also in this repo under the [./annotatedData](./annotatedData) folder. We are using a 80% training, 10% validation, 10% test split of the data. The test dataset is set up to be the data that was labeled by multiple independant labelers.
- Hugging Face
    - [Hugging Face](https://huggingface.co/) is a machine learning library, ML/AI community, and dedicated API that that is set up to assist with the creation, storage, and distribution of machine learning programs and datasets.
- Google Colab
    - [Google Colab](https://colab.research.google.com/) is a experimental framework for Jupiter Notebooks to be run on the cloud on GPUs and TPUs that are able to run independantly of your own computer's runtimes. This enables us to quickly build and test ML models using minimal resources.
- Weights and Biases
    - [Weights and Biases](https://wandb.ai/site) is a framework for tracking metrics and hyperparameters during training and evaluation of Machine Learning models. It assisted greatly with understanding where we could optimize our ML pipeline.
- Base Model Details
    - We primarilly make use of [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) and [bert-base-uncased](https://huggingface.co/bert-base-uncased) as the base models which we retrain for our specific use case. We use TinyBERT as the main model because it is only 59mb, which we deem as an acceptable size for achieving accuracy while remaining small enough to be loaded into a browser extension. We also explored [Knowledge Distillation](https://analyticsindiamag.com/a-beginners-guide-to-knowledge-distillation-in-deep-learning/) from bert-base-uncased to TinyBERT to better achieve accuracy at a smaller model size.
- How do the different scripts, frameworks, data interact?
    - Our datasets are located on the [privacy-tech-lab Hugging Face team](https://huggingface.co/privacy-tech-lab). Our files, scripts, and models are within the Privacy Pioneer Google Drive Folder's [Machine Learning](https://drive.google.com/drive/folders/1tjah6qy8JKf3RmI-ZxnKceiXuGahysE5?usp=sharing) section and also on Hugging Face.
- Where is the model you use in Privacy Pioneer?
    - The folder [./convertMultiModel/multitaskModelForJSWeb](./convertMultiModel/multitaskModelForJSWeb) is our model that we load into the Privacy Pioneer browser extension.

## 3. Links to Datasets and Final Models
- Datasets:
  - City:
    - [City Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppCityTrain)
    - [City Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppCityVal)
    - [City Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppCityTest)
  - Region:
    - [Region Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppRegionTrain)
    - [Region Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppRegionVal)
    - [Region Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppRegionTest)
  - Latitude:
    - [Lat Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppLatTrain)
    - [Lat Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppLatVal)
    - [Lat Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppLatTest)
  - Longitude:
    - [Lng Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppLngTrain)
    - [Lng Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppLngVal)
    - [Lng Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppLngTest)
  - Zip:
    - [Zip Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppZipTrain)
    - [Zip Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppZipVal)
    - [Zip Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppZipTest)
  - All:
    - [All Training Set](https://huggingface.co/datasets/privacy-tech-lab/ppAllTrain)
    - [All Validation Set](https://huggingface.co/datasets/privacy-tech-lab/ppAllVal)
    - [All Test Set](https://huggingface.co/datasets/privacy-tech-lab/ppAllTest)

- Models:
  - City:
    - [City TinyBERT Model](https://huggingface.co/privacy-tech-lab/CityModel)
    - [City bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/CityBaseModel)
    - [City Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/CityDistilledModel)
  - Region:
    - [Region TinyBERT Model](https://huggingface.co/privacy-tech-lab/RegionModel)
    - [Region bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/RegionBaseModel)
    - [Region Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/RegionDistilledModel)
  - Latitude:
    - [Lat TinyBERT Model](https://huggingface.co/privacy-tech-lab/LatModel)
    - [Lat bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/LatBaseModel)
    - [Lat Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/LatDistilledModel)
  - Longitude:
    - [Lng TinyBERT Model](https://huggingface.co/privacy-tech-lab/LngModel)
    - [Lng bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/LngBaseModel)
    - [Lng Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/LngDistilledModel)
  - Zip:
    - [Zip TinyBERT Model](https://huggingface.co/privacy-tech-lab/ZipModel)
    - [Zip bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/ZipBaseModel)
    - [Zip Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/ZipDistilledModel)
  - Multitask:
    - [Multitask TinyBERT Model](https://huggingface.co/privacy-tech-lab/MultitaskModel)
    - [Multitask bert-base-uncased Model](https://huggingface.co/privacy-tech-lab/MultitaskBaseModel)
    - [Multitask Base-Distilled-TinyBERT Model](https://huggingface.co/privacy-tech-lab/MultitaskDistilledModel)
    - [Multitask TinyBERT Model - tfjs Format](https://huggingface.co/privacy-tech-lab/multitaskModelJSWeb)

Results: (note: each value is the Average F1 Score)
<img src="./results.png">
## 4. Thank You!

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
