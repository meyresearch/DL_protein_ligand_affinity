# Role of proteins and ligands in deep learning-based binding affinity prediction

This repository contains the code and data for estimating protein-ligand binding affinity scores using various contact map techniques in the DL pipeline.

## Project Abstract

Protein-ligand binding affinity predictions are an important aspect of the early stages of drug discovery and recent deep learning-based methods have shown promise in predicting binding affinities. However, there is a lot of variability across these deep learning methods in terms of data encoding strategies for proteins and ligands which posed a challenge to understand the role played by encodings in the existing methods. Here, we investigate how different protein and ligand encodings will impact the binding affinity prediction and discover what the deep learning models are learning.

### Instructions to setup environment
- conda create --name mldd --file mldd1.txt 
- conda activate mldd
- pip install tensorflow-gpu

### Datasets and Models

Datasets and trained models can be downloaded [here](https://uoe-my.sharepoint.com/personal/s2112695_ed_ac_uk/_layouts/15/onedrive.aspx?login_hint=s2112695%40ed%2Eac%2Euk&id=%2Fpersonal%2Fs2112695%5Fed%5Fac%5Fuk%2FDocuments%2FBindingAffinity%5FDL%5FData)

### Steps for training

* Unzip the *data* folder from [here](https://uoe-my.sharepoint.com/personal/s2112695_ed_ac_uk/_layouts/15/onedrive.aspx?login_hint=s2112695%40ed%2Eac%2Euk&id=%2Fpersonal%2Fs2112695%5Fed%5Fac%5Fuk%2FDocuments%2FBindingAffinity%5FDL%5FData) and copy it to the main folder.
* Run the *Training_notebook.ipynb* file for training the graph-DL method using various protein and ligand encodings.



