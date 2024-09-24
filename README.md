# From Proteins to Ligands: Decoding Deep Learning Methods for Binding Affinity Prediction

This repository contains all code, data, instructions and model weights necessary to run or to retrain a model. 

## Project Abstract

Accurate *in silico* predictions of protein-ligand binding affinity can significantly accelerate the early stages of drug discovery. Deep learning-based methods have shown promise recently, but their robustness for the virtual screening of large compound libraries on various targets needs improvement. Understanding what these models learn from input protein and ligand data is essential to addressing this problem. We systematically investigated a sequence-based deep learning framework to assess the impact of protein and ligand encodings on commonly used kinase datasets. The role of proteins is studied using convolutional neural network-based encodings obtained from sequences and graph neural network-based encodings enriched with structural information from contact maps. By introducing perturbations to the ligand graph representation of the SMILES string, we assess the role played by ligand encodings given by the graph neural network. Our investigations show that protein encodings with structural information do not significantly impact the binding predictions, and the deep learning model relies heavily on ligand encodings for accurately predicting the binding affinity. Furthermore, various methods to combine protein and ligand encodings are explored, which showed no significant change in performance.

## Setup environment
We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the
current repo

    git clone https://github.com/meyresearch/Protein_ContactMaps_DL_BindingAffinity.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name mldd --file mldd.txt
    conda activate mldd
    pip install tensorflow-gpu


## Dataset

The files in `data` contain the datasets used in the study and the trained models.

If you want to train or test the models with the `data`  used in the study then: 
1. download it from [here](https://zenodo.org/records/13833868?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFjNjFmNWQzLWY0ODctNGNhNC1iNDJmLTI3YjI1N2JiMjc1OSIsImRhdGEiOnt9LCJyYW5kb20iOiIyNWI1ZDkyOGIzYTVlNzczNDc5MDk1MjZmNGIwZmJlZCJ9.MO1NkaIW1mWIcKwowa5lp_M3KSh1dQ3rJVHK5MFTlhaykYaPc-048faG1bUZvqBMtucEC-FGPE5k9oWFcNdpbA)
2. unzip the directory and place it into `data` such that you have the path `data/` in the main folder.


## Steps for training

* Unzip the `data` folder 
* Run the `Training_notebook.ipynb` file for training the DL method using various protein and ligand encodings.



