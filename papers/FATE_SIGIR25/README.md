# Fairness-aware Classification for Incomplete Data

This is the official implementation of our model FATE.

## Introduction

This folder (i.e., ./code) holds the source codes of the proposed system FATE
- args.py includes the arguments used in the experiments.
- dataloader.py addresses issues related to data loading.
- metrics.py contains various methods for measuring fairness and accuracy.
- model.py details the specifications and contents of the FATE model.
- run.py is the main file for running the FATE model.
- utils.py includes some specific utility functions and tools.

## Requirements
To install requirements:
```
pip install -r requirements.txt
```

## Data Description
All of the original datasets can refer to the links:
- Adult[1]: https://archive.ics.uci.edu/ml/datasets/adult 
- Compas[2]: https://www.kaggle.com/datasets/danofer/compass?select=cox-violent-parsed.csv
- HSLS[3]:  https://nces.ed.gov/EDAT/Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip
- ACS[4]:  https://www.census.gov/programs-surveys/acs

## Run the code
- Download the dataset and place it in the ./Datasets folder.
- You can use scripts in /scripts to run all the experiments.



## Reference
[1] M. Lichman. UCI machine learning repository, 2013.

[2] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias. ProPublica, 2016.

[3] Ingels, S. J., Pratt, D. J., Herget, D. R., Burns, L. J., Dever, J. A., Ottem, R., Rogers, J. E., Jin, Y., and Leinwand, S. (2011). High school longitudinal study of 2009 (hsls: 09): Base-year data file documentation. nces 2011-328. National Center for Education Statistics.

[4] Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. NeurIPS, 2021.