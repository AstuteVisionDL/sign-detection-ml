sign_detection_ml
==============================

Project to store shared code and notebooks with experiment

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
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and define them
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations for evaluation
    │   │
    │   └── config.py      <- config file for project
    │
    └── .pre-commit-config.yaml  <- pre-commit config file to run hooks on commit (run pre-commmit init to use)

