# Protein contact maps – how much do they matter in graph-based deep learning approaches for binding affinities?

This repository contains the code and data for estimating protein-ligand binding affinity scores using various contact map techniques in the DL pipeline.

## Project Abstract

Protein-ligand binding affinity (BA) predictions are an important aspect of the early stages of drug discovery and drug repurposing. Having reliable and transferable in silico methods will allow faster and cheaper routes to developing new drugs and reduce the need for costly experiments. Besides classical docking and alchemical free energy approaches, deep learning-based (DL) methods have shown promise in predicting binding affinities. These DL methods can be grouped into sequence-based and complex based approaches, while what we present here focuses on sequence-based affinity prediction methods. 
Earlier sequence-based approaches use protein sequences directly to embed primary structure information into vector space without considering the tertiary information which is important for understanding the protein-ligand interaction. 

DGraphDTA, a recent method, used tertiary information in the form of contact maps to construct protein graphs and outperformed earlier DL models on publicly available kinase datasets. Here, we investigate how different contact map prediction techniques will impact the overall binding affinity prediction in the DL pipeline. We study four distinct contact map prediction techniques- randomly generated contact maps that rely solely on sequence length, an unsupervised contact map prediction technique called ESM-1b which leverages sequence data only, and a supervised contact map prediction technique PconsC4 employs sequence information and homology data. The final technique generates contact maps from AlphaFold2 predicted 3D structures. 

### Instructions to setup environment
- conda create --name mldd --file mldd1.txt 
- conda activate mldd
- pip install tensorflow-gpu

### Datasets and Models

Datasets and trained models can be downloaded [here](https://uoe-my.sharepoint.com/personal/s2112695_ed_ac_uk/_layouts/15/onedrive.aspx?login_hint=s2112695%40ed%2Eac%2Euk&id=%2Fpersonal%2Fs2112695%5Fed%5Fac%5Fuk%2FDocuments%2FBindingAffinity%5FDL%5FData)

### Steps for testing and plotting

* Unzip the *data* folder from [here](https://uoe-my.sharepoint.com/personal/s2112695_ed_ac_uk/_layouts/15/onedrive.aspx?login_hint=s2112695%40ed%2Eac%2Euk&id=%2Fpersonal%2Fs2112695%5Fed%5Fac%5Fuk%2FDocuments%2FBindingAffinity%5FDL%5FData) and copy it to the DGraphDTA folder.
* Run the *KIBA_Results-Main.ipynb* file for testing using bootstrapping method and generating the resulting plots.

### Update 11th April 2022

- Code for contact map analysis project and results are added to the DGraphDTA folder. Here we are comparing various contact map tecniques in the DGraphDTA pipeline. To run the experiments setup a new environment with the mldd1.txt file in the DGraphDTA folder.


### Update 19th May 2022

- Organized the code to test and generate graphics.
