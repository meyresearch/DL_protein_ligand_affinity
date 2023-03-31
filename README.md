# From Proteins to Ligands: Decoding Deep Learning Methods for Binding Affinity Prediction

This repository contains all code, data, instructions and model weights necessary to run or to retrain a model. 

## Project Abstract

Protein-ligand binding affinity predictions are an essential aspect of the early stages of drug discovery. Deep learning-based methods recently have demonstrated promise in predicting protein-ligand affinities; however, the generalizability and deployability of these models for the virtual screening of large compound libraries on various targets still need improvement. Understanding what deep learning models are learning from the input protein and ligand data to predict binding affinity is key to addressing this problem. Our work investigates the impact of different protein and ligand encodings on binding affinity prediction and explores the role of protein structural information. Our results suggest that protein encodings do not significantly affect binding affinity prediction, and their structural information does not matter. In contrast, ligand encodings are crucial in the deep learning model's predictions. 


## Setup environment
We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the
current repo

    git clone https://github.com/meyresearch/Protein_ContactMaps_DL_BindingAffinity.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name mldd --file mldd1.txt
    conda activate mldd
    pip install tensorflow-gpu


## Dataset

The files in `data` contain the datasets used in the study and the trained models.

If you want to train or test the models with the `data`  used in the study then: 
1. download it from [here](https://uoe-my.sharepoint.com/personal/s2112695_ed_ac_uk/_layouts/15/onedrive.aspx?login_hint=s2112695%40ed%2Eac%2Euk&id=%2Fpersonal%2Fs2112695%5Fed%5Fac%5Fuk%2FDocuments%2FBindingAffinity%5FDL%5FData)
2. unzip the directory and place it into `data` such that you have the path `data/` in the main folder.


## Steps for training

* Unzip the `data` folder 
* Run the `Training_notebook.ipynb` file for training the DL method using various protein and ligand encodings.



