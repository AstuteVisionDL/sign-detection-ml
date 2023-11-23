sign_detection_ml
==============================

Project to store shared code and notebooks with experiment

# How to setup

1) Run clearml init
```bash
clearml-init
```
2) Get ClearML credentials. Open the ClearML Web UI in a browser. 
On the [SETTINGS > WORKSPACE](https://app.clear.ml/settings/workspace-configuration) page, click Create new credentials.
3) Enter the credentials in the terminal

# How to run

1) Run train.py script
```bash
python sign_recognition/train.py
```
*Note: You can change the config via command line arguments or by changing .yaml config in configs directory (https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)*

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make lint` or `make clean` - dev utils commands
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation for project in markdown format
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── pyproject.toml     <- The requirements and config file for reproducing the analysis environment and have config for project
    │
    ├── sign_recognition   <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate data, process it. Upload it to ClearML with new version
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling (Dataloaders, Datamodules, pytorch Dataset). Download data from ClearML
    │   │
    │   ├── models         <- Scripts to train models and define them
    │   │
    │   ├── envs.py        <- Environment variables for project
    │   │
    │   └── train.py       <- Main script to train models
    │
    └── .pre-commit-config.yaml  <- pre-commit config file to run hooks on commit (run pre-commmit init to use)

