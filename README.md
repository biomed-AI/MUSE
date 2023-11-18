# MUSE
Source code for "A Variational Expectation-Maximization Framework for Balanced Multi-scale Learning of Protein and Drug Interactions"

![workflow](https://github.com/biomed-AI/MUSE/blob/master/image/fig1.jpg)

# System requirement
MUSE is mainly based on the following packages:  
- python  3.9.16
- rdkit 2023.3.1
- numpy  1.24.3
- pytorch  1.12.1+cu116
- pytorch-scatter  1.6.1
- pytorch-cluster  2.1.1
- torch-sparse  0.6.17
- torch-geometric  2.3.1

Also you can install the required packages follow there instructions (tested on a linux terminal):

`conda env create -f environment.yaml`

`pip install -r requirements.txt`


# Dataset
We provide the datasets here for those interested in our paper:  
The protein and drug interactions datasets used in this study are stored in `./datasets/`


# Usage

## Protein and Drug Interaction Predictions
For EM training (for example, protein-protein interactions):
```
python trainer_ppi.py --cfg-path configs/ppi.yaml --job-id test_run
```

# Citation and contact
Citation: 
```
```  

Contact:  
Jiahua Rao (raojh6@mail2.sysu.edu.cn)
Yuedong Yang (yangyd25@mail.sysu.edu.cn)